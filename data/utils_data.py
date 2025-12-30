from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
import torch.nn.functional as F
import torch
from .data_splits import get_splits_in_domain, get_domain_splits
from torch_geometric.data import DataLoader
import numpy as np
import random


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     np.random.seed(seed)
#     random.seed(seed)


def COX2_MD_transform(data):
    # COX2,NCI
    # pad target dataset's x to [:,38]
    data.x = torch.cat([data.x, torch.zeros((data.x.shape[0], 38 - data.x.shape[1]))], dim=-1)
    return data


def BZR_MD_transform(data):
    # BZR,DHFR
    # pad target dataset's x to [:,56]
    data.x = torch.cat([data.x, torch.zeros((data.x.shape[0], 56 - data.x.shape[1]))], dim=-1)
    return data


def H_transform(data):
    # OVCAR-8,PC-3
    # pad only 1 zero to [:,1],use in source dataset
    data.x = torch.cat([data.x, torch.zeros((data.x.shape[0], 1))], dim=-1)
    return data


def PTC_transform(data):
    # OVCAR-8,PC-3
    data.x = torch.cat([data.x, torch.zeros((data.x.shape[0], 20 - data.x.shape[1]))], dim=-1)
    return data


def DD_transform(data):
    # DD
    # pad target dataset's x to [:,89]
    data.x = torch.cat([data.x, torch.zeros((data.x.shape[0], 89 - data.x.shape[1]))], dim=-1)
    return data


def no_node_label_transform(data):
    data.x = torch.ones((data.edge_index.max() + 1, 7))
    return data


two_dataset_mapping = {
    'COX2': 'COX2_MD',
    'BZR': 'BZR_MD',
    'NCI1': 'NCI109',
    'DHFR': 'DHFR_MD',
    'OVCAR-8': 'OVCAR-8H',
    'PC-3': 'PC-3H',
    'deezer_ego_nets': 'twitch_egos',
    'PTC_FM': 'PTC_FR',
    'PTC_MM': 'PTC_MR',
    'PROTEINS_full': 'DD'
}
multi_dataset_mapping = {
    'PTC': ['PTC_MR', 'PTC_MM', 'PTC_FM', 'PTC_FR']
}


def get_dataset(DS, path, args):
    # setup_seed(0)

    if args.cross_dataset == 1:  # cross dataset
        DSS = [DS, two_dataset_mapping[DS]]
        # load this 2 datasets as source and target
        source_split_index = args.source_index
        target_split_index = args.target_index
        source_dataset = TUDataset(path, name=DSS[source_split_index], use_node_attr=True, pre_transform=DD_transform)
        target_dataset = TUDataset(path, name=DSS[target_split_index], use_node_attr=True)
        split_dataset = [source_dataset, target_dataset]
        split_idx = [torch.arange(len(source_dataset)), torch.arange(len(target_dataset))]
        source_train_dataset, source_val_dataset, source_train_index, source_val_index = get_splits_in_domain(source_dataset, split_idx[0])
        target_train_dataset, target_test_dataset, target_train_index, target_test_index = get_splits_in_domain(target_dataset, split_idx[1])
        print(f'Dataset: {DSS}, Length: {len(source_dataset)},{len(target_dataset)}')
        print(f'Source Dim:{source_dataset[0].x.shape[1]}, Target Dim:{target_dataset[0].x.shape[1]}')
        return source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), (
        source_train_index, source_val_index, target_train_index, target_test_index), (DSS[source_split_index], DSS[target_split_index])
    elif args.cross_dataset == 2:  # cross domain
        DSS = multi_dataset_mapping[DS]
        # load this 2 datasets as source and target
        source_split_index = args.source_index
        target_split_index = args.target_index
        source_dataset = TUDataset(path, name=DSS[source_split_index], use_node_attr=True, pre_transform=PTC_transform)
        target_dataset = TUDataset(path, name=DSS[target_split_index], use_node_attr=True, pre_transform=PTC_transform)
        split_dataset = [source_dataset, target_dataset]
        split_idx = [torch.arange(len(source_dataset)), torch.arange(len(target_dataset))]
        source_train_dataset, source_val_dataset, source_train_index, source_val_index = get_splits_in_domain(source_dataset, split_idx[0])
        target_train_dataset, target_test_dataset, target_train_index, target_test_index = get_splits_in_domain(target_dataset, split_idx[1])
        print(f'Dataset: {DSS}, Length: {len(source_dataset)},{len(target_dataset)}')
        print(f'Source Dim:{source_dataset[0].x.shape[1]}, Target Dim:{target_dataset[0].x.shape[1]}')
        return source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), (
        source_train_index, source_val_index, target_train_index, target_test_index), (DSS[source_split_index], DSS[target_split_index])
    else:
        dataset = TUDataset(path, name=DS, use_node_attr=True)
        print(f'Dataset: {DS}, Length: {len(dataset)}')
        source_split_index = args.source_index
        target_split_index = args.target_index
        split = args.data_split
        split_dataset, split_idx = get_domain_splits(dataset, split)
        source_dataset = split_dataset[source_split_index]
        target_dataset = split_dataset[target_split_index]
        source_train_dataset, source_val_dataset, source_train_index, source_val_index = get_splits_in_domain(source_dataset,
                                                                                                              split_idx[source_split_index])
        target_train_dataset, target_test_dataset, target_train_index, target_test_index = get_splits_in_domain(target_dataset,
                                                                                                                split_idx[target_split_index])
    # sum target_test_dataset'y
    print("True %:", 100.0 * sum(target_test_dataset.data.y) / len(target_test_dataset.data.y))
    return source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), (
    source_train_index, source_val_index, target_train_index, target_test_index), (DS, DS)


def split_confident_data_old(model, dataset, saved_index, device, args):
    model.eval()
    confident_percentage = args.confident_rate
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    confident_dataset = []
    inconfident_dataset = []
    confident_idx = []
    inconfident_idx = []
    correct_count = 0
    for data in loader:
        data = data.to(device)
        c, n, z = model(data.x, data.edge_index, data.batch, data.num_graphs)
        prob = torch.softmax(model.classifier(c), dim=-1)
        confident_id = prob.max(dim=-1)[0].topk(int(len(prob) * confident_percentage))[1]
        confident_mask = torch.zeros(len(prob), dtype=torch.bool).to(device)
        confident_mask[confident_id] = True
        inconfident_id = torch.where(~confident_mask)[0]
        confident_dataset += data[confident_mask]
        inconfident_dataset += data[~confident_mask]
        confident_idx += saved_index[confident_id].tolist()
        inconfident_idx += saved_index[inconfident_id].tolist()
        correct_count += (prob.max(dim=-1)[1] == data.y).sum().item()

    train_ratio = 0.8
    confident_train_dataset, confident_val_dataset = confident_dataset[:int(len(confident_dataset) * train_ratio)], confident_dataset[int(len(
        confident_dataset) * train_ratio):]
    confident_train_idx, confident_val_idx = confident_idx[:int(len(confident_idx) * train_ratio)], confident_idx[
                                                                                                    int(len(confident_idx) * train_ratio):]
    # check correct rate of confident_train_dataset
    correct_count = 0
    confident_dataloader = DataLoader(confident_train_dataset, batch_size=2048, shuffle=False)
    for data in confident_dataloader:
        data = data.to(device)
        c, n, z = model(data.x, data.edge_index, data.batch, data.num_graphs)
        prob = torch.softmax(model.classifier(c), dim=-1)
        correct_count += (prob.max(dim=-1)[1] == data.y).sum().item()
    print("Confident Train Correct Rate(in utils_data.py):", correct_count / len(confident_train_dataset), "correct count:", correct_count,
          "total count:", len(confident_train_dataset))
    # print(confident_train_dataset)
    return confident_train_dataset, confident_val_dataset, inconfident_dataset, (confident_train_idx, confident_val_idx, inconfident_idx)


def prediction_entropy(probs: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """
    计算一批样本的预测熵。

    Args:
        probs (torch.Tensor): 模型的输出概率分布，形状为 [batch_size, num_classes]。
                              必须是经过 Softmax 或其他归一化操作后的概率。
        epsilon (float): 一个极小的数，用于防止 log(0) 的计算错误。

    Returns:
        torch.Tensor: 每个样本的预测熵，形状为 [batch_size]。
    """
    # 确保输入是概率分布（每行和为1）
    # assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0)), "Input tensor must be a probability distribution."

    # 为概率值添加一个极小的数，避免 log(0)
    probs = probs + epsilon

    # 计算熵: H(p) = - Σ (p * log(p))
    # log_probs 的形状是 [batch_size, num_classes]
    log_probs = torch.log(probs)

    # entropy 的形状是 [batch_size]
    entropy = -torch.sum(probs * log_probs, dim=1)

    return entropy


def split_confident_data(model, diffusion_model, dataset, saved_index, device, args, target_domain):
    model.eval()
    diffusion_model.model.eval()
    diffusion_model.fp_encoder.eval()

    loader = DataLoader(dataset, batch_size=2048, shuffle=False)

    all_raw_probs = []
    all_denoised_probs = []

    print("Generating raw and denoised pseudo-labels...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # 获取原始伪标签概率
            c, n, z = model(data.x, data.edge_index, data.batch, data.num_graphs, domain=target_domain)
            raw_probs = torch.softmax(model.classifier(c), dim=-1)
            all_raw_probs.append(raw_probs)

            # 使用diffusion model降噪
            denoised_probs = diffusion_model.reverse_ddim(data, stochastic=False, fq_x=None, domain=target_domain)
            denoised_probs = torch.softmax(-(denoised_probs - 1) ** 2, dim=-1)
            all_denoised_probs.append(denoised_probs)

    all_raw_probs = torch.cat(all_raw_probs, dim=0)
    all_denoised_probs = torch.cat(all_denoised_probs, dim=0)

    # 计算一致性损失 symmetrized KL divergence
    epsilon = 1e-12
    log_denoised = torch.log(all_denoised_probs + epsilon)
    log_raw = torch.log(all_raw_probs + epsilon)
    loss_kl_1 = F.kl_div(log_denoised, all_raw_probs, reduction='none').sum(dim=1)
    loss_kl_2 = F.kl_div(log_raw, all_denoised_probs, reduction='none').sum(dim=1)
    consistency_loss = (loss_kl_1 + loss_kl_2) / 2.0
    predict_entropy = prediction_entropy(all_raw_probs)
    normalized_consistency = consistency_loss / (consistency_loss.max() + 1e-8)
    normalized_entropy = predict_entropy / (predict_entropy.max() + 1e-8)
    combined_score = normalized_consistency + 0.5 * normalized_entropy

    # 根据一致性损失选择置信样本
    num_confident = int(len(dataset) * args.confident_threshold)
    _, confident_id = torch.topk(combined_score, k=num_confident, largest=False)
    confident_mask = torch.zeros(len(dataset), dtype=torch.bool, device=device)
    confident_mask[confident_id] = True

    # 划分数据集
    confident_dataset = dataset[confident_mask]
    inconfident_dataset = dataset[~confident_mask]

    confident_idx = saved_index[confident_mask].tolist()
    inconfident_idx = saved_index[~confident_mask].tolist()

    print(f"Total target samples: {len(dataset)}")
    print(f"Selected {len(confident_dataset)} confident samples based on diffusion consistency.")

    # 检查置信样本的伪标签准确率 (用于分析，非训练必须)
    correct_count = 0
    confident_dataloader = DataLoader(confident_dataset, batch_size=2048, shuffle=False)
    with torch.no_grad():
        for data in confident_dataloader:
            data = data.to(device)
            # 使用原始伪标签的argmax作为预测
            c, _, _ = model(data.x, data.edge_index, data.batch, data.num_graphs, domain=target_domain)
            pred = model.classifier(c).argmax(dim=-1)
            correct_count += (pred == data.y).sum().item()
    pseudo_label_accuracy = correct_count / len(confident_dataset) if len(confident_dataset) > 0 else 0
    print(f"Confident Set Pseudo-Label Accuracy: {pseudo_label_accuracy * 100:.2f}%")

    # 划分训练集和验证集
    train_ratio = 0.8
    num_train = int(len(confident_dataset) * train_ratio)

    confident_train_dataset = confident_dataset[:num_train]
    confident_val_dataset = confident_dataset[num_train:]

    confident_train_idx = confident_idx[:num_train]
    confident_val_idx = confident_idx[num_train:]

    return confident_train_dataset, confident_val_dataset, inconfident_dataset, (confident_train_idx, confident_val_idx, inconfident_idx)


    # confident_threshold = args.confident_threshold
    # loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    # confident_dataset = []
    # inconfident_dataset = []
    # confident_idx = []
    # inconfident_idx = []
    # correct_count = 0
    # for data in loader:
    #     data = data.to(device)
    #     c, n, z = model(data.x, data.edge_index, data.batch, data.num_graphs)
    #     prob = torch.softmax(model.classifier(c), dim=-1)
    #     # confident_id = prob.max(dim=-1)[0].topk(int(len(prob)*confident_percentage))[1]
    #     confident_id = []
    #     for label in range(prob.shape[1]):
    #         maxoflabel = prob[:, label].max()
    #         label_threshold = maxoflabel * confident_threshold
    #         confident_id_label = torch.where(prob[:, label] > label_threshold)[0]
    #         confident_id.append(confident_id_label)
    #     confident_id = torch.cat(confident_id)
    #     confident_mask = torch.zeros(len(prob), dtype=torch.bool).to(device)
    #     confident_mask[confident_id] = True
    #     inconfident_id = torch.where(~confident_mask)[0]
    #     confident_dataset += data[confident_mask]
    #     inconfident_dataset += data[~confident_mask]
    #     confident_idx += saved_index[confident_id].tolist()
    #     inconfident_idx += saved_index[inconfident_id].tolist()
    #     correct_count += (prob.max(dim=-1)[1] == data.y).sum().item()
    #
    # train_ratio = 0.8
    # confident_train_dataset, confident_val_dataset = confident_dataset[:int(len(confident_dataset) * train_ratio)], confident_dataset[int(len(
    #     confident_dataset) * train_ratio):]
    # confident_train_idx, confident_val_idx = confident_idx[:int(len(confident_idx) * train_ratio)], confident_idx[
    #                                                                                                 int(len(confident_idx) * train_ratio):]
    # # check correct rate of confident_train_dataset
    # correct_count = 0
    # confident_dataloader = DataLoader(confident_train_dataset, batch_size=2048, shuffle=False)
    # for data in confident_dataloader:
    #     data = data.to(device)
    #     c, n, z = model(data.x, data.edge_index, data.batch, data.num_graphs)
    #     prob = torch.softmax(model.classifier(c), dim=-1)
    #     correct_count += (prob.max(dim=-1)[1] == data.y).sum().item()
    # print("Confident Train Correct Rate(in utils_data.py):", correct_count / len(confident_train_dataset), "correct count:", correct_count,
    #       "total count:", len(confident_train_dataset))
    # # print(confident_train_dataset)
    # return confident_train_dataset, confident_val_dataset, inconfident_dataset, (confident_train_idx, confident_val_idx, inconfident_idx)
