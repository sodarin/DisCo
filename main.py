import itertools
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
import time
from arguments import arg_parse
from data.utils_data import get_dataset, split_confident_data
from data.graph_aug import AugTransform, NoiseTransform
from torch.utils.data import SequentialSampler

from diffusion.train import train_diffusion
from models.model import GNN
from torch.autograd import Function
from utils.utils import compute_mmd_batch, get_logger, setup_seed
import copy
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import add_self_loops, to_dense_adj, dense_to_sparse
from utils.mixup_utils import prepare_graphon, prepare_aligned_dataset, prepare_augmented_dataset


def count_parameters(model: torch.nn.Module):
    """
    计算并打印 PyTorch 模型的总参数量和可训练参数量。

    Args:
        model (torch.nn.Module): 需要计算参数量的模型。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    return total_params, trainable_params


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


def orthogonality_loss(z_c, z_s):
    # 中心化特征
    z_c = z_c - z_c.mean(dim=0)
    z_s = z_s - z_s.mean(dim=0)

    # 计算批次内的协方差矩阵
    cov_matrix = torch.matmul(z_c.T, z_s) / (z_c.shape[0] - 1)

    # 损失是协方差矩阵的Frobenius范数的平方
    loss = torch.sum(cov_matrix ** 2)
    return loss


def T_c(f_c, y, W_c):
    y = y.unsqueeze(-1).float()
    return torch.mm(f_c, torch.mm(W_c, y))


def T_n(f_n, z, W_n):
    return torch.mm(f_n, torch.mm(W_n, z.T))


def causal_loss(f_c, y, W_c, temperature=0.1):
    z = F.normalize(f_c, dim=1)
    scores = torch.matmul(z, z.t()) / temperature  # [256, 256]
    positive_samples = torch.diag(scores)
    negative_scores = torch.logsumexp(scores, dim=1, keepdim=False)
    loss = torch.mean(negative_scores - positive_samples)
    return loss


def non_causal_loss(f_n, y, z, W_n, beta):
    log_p_y_given_fn = F.log_softmax(f_n, dim=1)
    true_class_log_probs = log_p_y_given_fn.gather(1, y.view(-1, 1)).squeeze()
    T_n_fn_z = f_n
    logsumexp_T_n_fn_z = torch.logsumexp(T_n_fn_z, dim=1)

    return torch.mean(true_class_log_probs) - beta * torch.mean(logsumexp_T_n_fn_z)


class GraphGenerator(torch.nn.Module):
    def __init__(self, num_features):
        super(GraphGenerator, self).__init__()
        self.fc1 = torch.nn.Linear(num_features * 2, num_features)
        self.fc2 = torch.nn.Linear(num_features, num_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@torch.no_grad()
def test(loader, model, domain=None):
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        c, n, z = model(data.x, data.edge_index, data.batch, data.num_graphs, domain=domain)
        pred = model.classifier(c).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


@torch.no_grad()
def eval_train(loader, model, domain=None):
    model.eval()
    total_correct = 0
    for data_dict in loader:
        data = data_dict.to(device)
        c, n, z = model(data.x, data.edge_index, data.batch, data.num_graphs, domain=domain)
        pred = model.classifier(c).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


@torch.no_grad()
def eval_train_motif(loader, model):
    model.eval()
    total_correct = 0
    for data_dict in loader:
        motif0 = data_dict[0]
        motif0 = motif0.to(device)
        motif_out, motif_embedding = model(data_dict)
        pred = motif_out.argmax(dim=-1)
        total_correct += int((pred == motif0.y).sum())
    return total_correct / len(loader.dataset)


def run(seed):
    logger.info('seed:{}'.format(seed))
    setup_seed(seed)

    epochs = args.epochs
    eval_interval = args.eval_interval
    sdfa_eval_interval = args.sdfa_eval_interval
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    source_domain = None
    target_domain = None

    if args.cross_dataset == 1:
        source_domain = 'source'
        target_domain = 'target'

    source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), \
    (source_train_index, source_val_index, target_train_index, target_test_index), (sDS, tDS) = get_dataset(DS, path, args)

    print((source_train_index.shape, source_val_index.shape, target_train_index.shape, target_test_index.shape))

    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_transforms = NoiseTransform()

    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    nll_criterion = nn.NLLLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')

    args.dataset_num_features = source_train_dataset[0].x.shape[1]
    args.source_dataset_num_features = args.dataset_num_features
    args.target_dataset_num_features = target_train_dataset[0].x.shape[1]

    model = GNN(args.dataset_num_features, args.hidden_dim, args.num_gc_layers, source_dataset.num_classes, args, device).to(device)
    generator = GraphGenerator(args.hidden_dim).to(device)

    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-2)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=5e-2)

    print(f'=======[Source: {args.source_index} -> Target: {args.target_index}]=========')

    # Train on Source Domain 
    best_val_acc = 0.0
    final_test_acc = 0.0
    best_model = copy.deepcopy(model)

    source_time_start = time.time()
    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()
        dataloader = source_train_loader

        for data_dict in dataloader:
            data = data_dict.to(device)
            optimizer.zero_grad()

            x_c, x_n, x_z = model(data.x, data.edge_index, data.batch, data.num_graphs, domain=source_domain)

            # y = torch.tensor([1] * len(data.y)).to(device)
            # loss_causal = causal_loss(x_c, y, model.W_c)
            # loss_non_causal = non_causal_loss(x_n, y, x_z, model.W_n, 0.1)
            # loss_IB = (loss_causal - loss_non_causal)

            # x_cat = torch.cat((x_c, x_n), dim=1)
            # x_z_pred = generator(x_cat)
            # loss_consistency = kl_criterion(F.log_softmax(x_z, dim=1), F.softmax(x_z_pred, dim=1)) * 0.5
            pred = model.classifier(x_c)
            loss_cls = ce_criterion(pred, data.y)
            loss = loss_cls
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        if epoch % eval_interval == 0:
            model.eval()
            train_acc = eval_train(dataloader, model, domain=source_domain)
            val_acc = test(source_val_loader, model, domain=source_domain)
            test_acc = test(target_test_loader, model, domain=target_domain)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                raw_target_acc = test_acc
                best_model = copy.deepcopy(model)
            print(f'Epoch: {epoch:03d}, Loss: {loss_all / len(dataloader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f} ')
            # print(f'Cls: {loss_cls:.4f}, Consistency: {loss_consistency:.4f}, IB: {loss_IB:.4f}')

    source_time_end = time.time()
    print(f'Source training time: {source_time_end - source_time_start}')
    model = best_model
    pre_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predict_labels = []
    for data in pre_loader:
        data = data.to(device)
        model.eval()
        x_c, _, _ = model(data.x, data.edge_index, data.batch, data.num_graphs, domain=source_domain)
        pred = model.classifier(x_c)
        predict_labels.append(pred)
    predict_labels = torch.vstack(predict_labels)

    diff_time_start = time.time()
    diffusion_model = train_diffusion(args, model, source_train_dataset, predict_labels)
    count_parameters(diffusion_model)
    diff_time_end = time.time()
    print(f'Diffusion training time: {diff_time_end - diff_time_start}')

    confident_train_dataset, confident_val_dataset, inconfident_dataset, (
        confident_train_index, confident_val_index, inconfident_index) = split_confident_data(model, diffusion_model, target_train_dataset,
                                                                                              target_train_index, device,
                                                                                              args, target_domain)
    print(f'confident_data:{len(confident_train_dataset)},inconfident_data:{len(inconfident_dataset)}')

    source_train_index = torch.randperm(min(len(source_train_dataset), len(target_train_dataset)))
    target_train_index = torch.randperm(min(len(source_train_dataset), len(target_train_dataset)))
    source_train_sampler = SequentialSampler(source_train_index)
    target_train_sampler = SequentialSampler(target_train_index)
    source_train_dataloader = DataLoader(source_train_dataset, batch_size=batch_size, sampler=source_train_sampler, num_workers=0)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=batch_size, sampler=target_train_sampler, num_workers=0)

    train_index = torch.randperm(len(confident_train_dataset))
    val_index = torch.randperm(len(confident_val_dataset))
    train_sampler = SequentialSampler(train_index)
    val_sampler = SequentialSampler(val_index)
    confident_train_dataloder = DataLoader(confident_train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    confident_val_dataloder = DataLoader(confident_val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)

    model_for_pseudo_label = copy.deepcopy(model)
    model_for_pseudo_label.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.sdfa_lr, weight_decay=5e-3)

    target_epochs = args.target_epochs
    best_val_acc = raw_target_acc
    final_test_acc = raw_target_acc
    start_t = time.time()
    if args.causal:
        for epoch in range(1, target_epochs + 1):
            model.train()
            for batch_s, batch_t, batch_con in zip(source_train_dataloader, target_train_dataloader, confident_train_dataloder):
                batch_s, batch_t = batch_s.to(device), batch_t.to(device)
                batch_con = batch_con.to(device)
                y = torch.tensor([0] * len(batch_s.y) + [1] * len(batch_t.y)).to(device)
                optimizer.zero_grad()
                x_s_c, x_s_n, x_s_z = model(batch_s.x, batch_s.edge_index, batch_s.batch, batch_s.num_graphs, domain=source_domain)
                x_t_c, x_t_n, x_t_z = model(batch_t.x, batch_t.edge_index, batch_t.batch, batch_t.num_graphs, domain=target_domain)

                f_c = torch.cat((x_s_c, x_t_c), dim=0)
                f_n = torch.cat((x_s_n, x_t_n), dim=0)
                z = torch.cat((x_s_z, x_t_z), dim=0)

                domain_labels = torch.cat([torch.zeros(x_s_c.size(0), dtype=torch.long, device=device),
                                           torch.ones(x_t_c.size(0), dtype=torch.long, device=device)])

                # causal adversarial loss
                reversed_f_c = grad_reverse(f_c, alpha=args.reverse_alpha)
                domain_pred_c = model.domain_discriminator(reversed_f_c)
                loss_adv_causal = ce_criterion(domain_pred_c, domain_labels)

                # spurious discriminative loss
                domain_pred_n = model.domain_discriminator(f_n)
                loss_adv_spurious = ce_criterion(domain_pred_n, domain_labels)

                # 正交损失
                loss_ortho = orthogonality_loss(f_c, f_n)

                # loss_causal = causal_loss(f_c, y, model.W_c)
                # loss_non_causal = non_causal_loss(f_n, y, z, model.W_n, 0.1)
                # loss_IB = (loss_causal - loss_non_causal)

                x_c_c, x_c_n, x_c_z = model(batch_con.x, batch_con.edge_index, batch_con.batch, batch_con.num_graphs, domain=target_domain)
                pred_c = model.classifier(x_c_c)
                pred_s = model.classifier(x_s_c)
                with torch.no_grad():
                    x_p_c, x_p_n, x_p_z = model_for_pseudo_label(batch_con.x, batch_con.edge_index, batch_con.batch, batch_con.num_graphs,
                                                                 domain=target_domain)
                    pred_p = model_for_pseudo_label.classifier(x_p_c)
                    pseudo_label = pred_p.argmax(dim=-1)
                loss_dis = ce_criterion(pred_c, pseudo_label)
                loss_source = ce_criterion(pred_s, batch_s.y)
                # f_n_shuffled = f_n[torch.randperm(f_n.shape[0])]
                # z_aug = generator(torch.cat((f_c, f_n_shuffled), dim=1))
                # loss_inv = mse_criterion(z, z_aug) * 0.02

                # total_loss = loss_IB * args.IB_weight + loss_dis * args.DIS_weight + loss_inv * args.INV_weight + loss_source * args.source_weight

                total_loss = (loss_adv_spurious + loss_adv_causal) * args.IB_weight + loss_dis * args.DIS_weight + \
                             loss_ortho * args.INV_weight + loss_source * args.source_weight
                print(
                    f'Epoch: {epoch:03d}, Loss: {total_loss:.2f}, Causal: {loss_adv_causal:.4f}, Non-Causal: {loss_adv_spurious:.4f}, '
                    f'Discriminative: {loss_dis:.4f}, Orthogonality: {loss_ortho:.4f}, Source: {loss_source:.4f}')
                total_loss.backward()
                optimizer.step()

            if epoch % sdfa_eval_interval == 0:
                model.eval()
                train_acc = eval_train(confident_train_dataloder, model, domain=target_domain)
                val_acc = test(confident_val_dataloder, model, domain=target_domain)
                test_acc = test(target_test_loader, model, domain=target_domain)
                if test_acc > final_test_acc:
                    final_test_acc = test_acc
                print(
                    f'SFDA Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                    f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    end_t = time.time()
    print(f'Adaptation time: {end_t - start_t}')
    logger.info('raw_target_acc: {:.2f}, final_target_acc: {:.2f}'.format(raw_target_acc * 100, final_test_acc * 100))

    return raw_target_acc, final_test_acc


if __name__ == '__main__':
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logger = get_logger(args)

    # IB_weight_list = [0.003, 0.01, 0.1, 0.5]
    # INV_weight_list = [0.05, 0.1, 0.5, 1.0]
    # DIS_weight_list = [0.1, 0.5, 1.0]
    # source_weight_list = [0.1, 0.5, 1.0]
    # confident_threshold_list = [0.05, 0.1]
    # reverse_alpha_list = [0.1, 0.5, 1.0]

    IB_weight_list = [0.01]
    INV_weight_list = [0.05]
    DIS_weight_list = [0.1]
    source_weight_list = [0.5]
    confident_threshold_list = [0.1]
    reverse_alpha_list = [0.1]

    for args.IB_weight, args.INV_weight, args.DIS_weight, args.source_weight, args.confident_threshold, args.reverse_alpha in \
            itertools.product(IB_weight_list, INV_weight_list, DIS_weight_list, source_weight_list, confident_threshold_list, reverse_alpha_list):
        logger.info(
            f'IB_weight: {args.IB_weight}, INV_weight: {args.INV_weight}, DIS_weight: {args.DIS_weight}, source_weight: {args.source_weight}, \n'
            f'confident_threshold: {args.confident_threshold}, reverse_alpha: {args.reverse_alpha}')
        for args.st_seed in range(0, 3):
            seed = args.st_seed
            acc_list = []
            raw_acc_list = []
            for i in range(args.st_seed, args.st_seed + args.number_of_run):
                raw_acc, test_acc = run(seed)
                acc_list.append(test_acc)
                raw_acc_list.append(raw_acc)

            print(acc_list)
            acc_mean = np.mean(acc_list)
            acc_std = np.std(acc_list)
            print("final_acc_mean: {:.2f}, final_acc_std: {:.2f}".format(acc_mean * 100, acc_std * 100))
            raw_acc_mean = np.mean(raw_acc_list)
            raw_acc_std = np.std(raw_acc_list)
            print("raw_acc_mean: {:.2f}, raw_acc_std: {:.2f}".format(raw_acc_mean * 100, raw_acc_std * 100))

            logger.info("raw_acc_mean: {:.2f}, raw_acc_std: {:.2f}".format(raw_acc_mean * 100, raw_acc_std * 100))
            logger.info("final_acc_mean: {:.2f}, final_acc_std: {:.2f}".format(acc_mean * 100, acc_std * 100))

