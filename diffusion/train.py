import copy
import shutil
import tempfile
import torch.nn as nn
import torch.utils.data as data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from diffusion.ema import EMA
import numpy as np
import random
import time
import torch.optim as optim
from diffusion.learning import *
from diffusion.model_diffusion import Diffusion
from diffusion.knn_utils import sample_knn_labels
import os
import argparse
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
# np.random.seed(123)
# random.seed(123)


def train(diffusion_model, train_dataset, val_dataset, model_path, args, real_fp):
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs

    source_domain = None
    target_domain = None

    if args.cross_dataset == 1:
        source_domain = 'source'
        target_domain = 'target'

    # # pre-compute for fp embeddings on training data
    print('pre-computing fp embeddings for training data')
    train_embed = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, save_dir=None, device=device,
                               fp_dim=args.hidden_dim, domain=source_domain).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.diffusion_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.diffusion_batch_size, shuffle=False, num_workers=0)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy = 0.0
    best_model = None
    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                data_batch.to(device)

                fp_embd, _, _ = diffusion_model.fp_encoder(data_batch.x, data_batch.edge_index, data_batch.batch, data_batch.num_graphs,
                                                           source_domain)

                # sample a knn labels and compute weight for the sample
                y_labels_batch, sample_weight = sample_knn_labels(fp_embd, data_batch.y.to(device), train_embed,
                                                                  torch.tensor(train_dataset.y).to(device),
                                                                  k=k, n_class=n_class, weighted=True)

                # convert label to one-hot vector
                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.to(torch.int64),
                                                                                      n_class=n_class)
                y_0_batch = y_one_hot_batch.to(device)

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.001)
                n = data_batch.num_graphs

                # sampling t
                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                # train with and without prior
                output, e = diffusion_model.forward_t(y_0_batch, data_batch, t, fp_embd, domain=source_domain)

                # compute loss
                mse_loss = diffusion_loss(e, output)
                weighted_mse_loss = torch.matmul(sample_weight, mse_loss)
                loss = torch.mean(weighted_mse_loss)
                pbar.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        if epoch % 5 == 0 and epoch >= warmup_epochs:
            val_acc = test(args, diffusion_model, val_loader, source_domain)
            print(f"epoch: {epoch}, validation accuracy: {val_acc:.2f}%")
            if val_acc > max_accuracy:
                # save diffusion model
                print('Improved! evaluate on testing set...')
                states = [diffusion_model.model.state_dict(),
                          diffusion_model.diffusion_encoder.state_dict(),
                          diffusion_model.fp_encoder.state_dict()]
                best_model = diffusion_model
                torch.save(states, model_path)
                print(f"Model saved, update best accuracy at Epoch {epoch}, val acc: {val_acc}")
                max_accuracy = max(max_accuracy, val_acc)

    return best_model


def test(args, diffusion_model, test_loader, source_domain):
    start = time.time()
    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0
        all_cnt = 0
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            target = data_batch.y
            label_t_0 = diffusion_model.reverse_ddim(data_batch, stochastic=False, fq_x=None, domain=source_domain).detach().cpu()
            correct = cnt_agree(label_t_0.detach().cpu(), target)
            correct_cnt += correct
            all_cnt += data_batch.num_graphs

    print(f'time cost for CLR: {time.time() - start}')

    acc = 100 * correct_cnt / all_cnt
    return acc


def train_diffusion(args, fp_encoder, dataset, predict_labels):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    # parser.add_argument("--batch_size", default=200, help="batch_size", type=int)
    # parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    # parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    # parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    # parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    # parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    # parser.add_argument("--CLIP_type", default='ViT-L/14', help="which encoder for CLIP", type=str)
    # parser.add_argument("--diff_encoder", default='resnet34', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    # args = parser.parse_args()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Using device:', device)

    n_class = predict_labels.size(1)
    fp_dim = args.hidden_dim

    # initialize diffusion model
    script_dr = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dr, 'parameters', f'LRA-diffusion_{args.DS}_{args.source_index}_{args.target_index}.pt')
    diffusion_model = Diffusion(args=args, fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=fp_dim,
                                device=device, ddim_num_steps=args.ddim_n_step)

    diffusion_model.fp_encoder.eval()

    data_idx = torch.randperm(len(dataset))
    train_split_idx = int(len(data_idx) * 0.8)

    train_indices = data_idx[:train_split_idx]
    val_indices = data_idx[train_split_idx:]

    train_predict_labels = predict_labels[train_indices]
    val_predict_labels = predict_labels[val_indices]

    train_data_list = create_new_dataset(dataset, train_indices, train_predict_labels)
    val_data_list = create_new_dataset(dataset, val_indices, val_predict_labels)

    train_root = script_dr + '/data_temp'
    val_root = script_dr + '/val_data_temp'

    if os.path.exists(train_root) and os.path.exists(val_root):
        shutil.rmtree(train_root)
        shutil.rmtree(val_root)

    train_dataset = MyCustomDataset(root=train_root, data_list=train_data_list)
    val_dataset = MyCustomDataset(root=val_root, data_list=val_data_list)

    # train the diffusion model
    print(f'diffusion model saving dir: {model_path}')
    model = train(diffusion_model, train_dataset, val_dataset, model_path, args, real_fp=True)

    return model


def create_new_dataset(original_dataset, indices, new_labels):
    """
    根据给定的索引和新标签，从原始数据集中创建一份新的Data对象列表。
    """
    new_data_list = []
    # 注意：这里的 new_labels 和 indices 是一一对应的
    for i, idx in enumerate(indices):
        # 从原始数据集中获取图数据
        original_data = original_dataset[idx]

        # 克隆数据以避免修改原始数据集
        new_data = original_data.clone()

        # 将新的预测标签赋值给 y 属性
        # new_labels[i] 是一个张量，我们直接赋值
        new_data.y = torch.argmax(new_labels[i].detach().cpu(), dim=-1).unsqueeze(0)

        new_data_list.append(new_data)

    return new_data_list


# 5. 为 PyG 的 InMemoryDataset 创建一个简单的包装类
#    这确保了数据集的行为和 PyG 的生态系统完全兼容
class MyCustomDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        # 我们将 data_list 存储起来，以便在 process 方法中使用
        self.data_list = data_list
        super().__init__(root, transform)
        # 关键：__init__ 方法会调用 process（如果需要），然后我们从已处理的文件中加载数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 因为我们直接提供了 data_list，所以没有原始文件
        return []

    @property
    def processed_file_names(self):
        # 定义处理后的数据将存储在哪个文件中
        return ['data.pt']

    def download(self):
        # 不需要下载
        pass

    def process(self):
        # 这是核心。如果 processed_paths[0] 文件不存在，此方法将被调用。
        # 它将我们的 Python Data 对象列表转换为 PyG 内部使用的大型图表示，并保存到磁盘。
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])
