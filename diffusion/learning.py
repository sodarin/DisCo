import random
import math
import numpy as np
import torch
from torch import nn
import os
import torch.utils.data as data
from tqdm import tqdm
from torch_geometric.loader import DataLoader


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch, warmup_epochs=40, n_epochs=1000, lr_input=0.001):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr_input * epoch / warmup_epochs
    else:
        lr = 0.0 + lr_input * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def cast_label_to_one_hot_and_prototype(y_labels_batch, n_class, return_prototype=True):
    """
    y_labels_batch: a vector of length batch_size.
    """
    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=n_class).float()
    if return_prototype:
        label_min, label_max = [0.001, 0.999]
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch


# random seed related
def init_fn(worker_id):
    np.random.seed(77 + worker_id)


def prepare_fp_x(fp_encoder, dataset, save_dir=None, device='cpu', fp_dim=768, batch_size=400, domain=None):
    if save_dir is not None:
        if os.path.exists(save_dir):
            fp_embed_all = torch.tensor(np.load(save_dir))
            print(f'Embeddings were computed before, loaded from: {save_dir}')
            return fp_embed_all.cpu()

    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        fp_embed_all = []
        for data_batch in data_loader:
            data_batch.to(device)
            temp, _, _ = fp_encoder(data_batch.x, data_batch.edge_index, data_batch.batch, data_batch.num_graphs, domain)
            fp_embed_all.append(temp.detach().cpu())
        # fp_embed_all = np.array(fp_embed_all)
        fp_embed_all = torch.concat(fp_embed_all, dim=0).squeeze(0)
        if save_dir is not None:
            np.save(save_dir, fp_embed_all)

    return fp_embed_all


def cnt_agree(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.size()[1])

    output = torch.softmax(-(output - 1) ** 2, dim=-1)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return torch.sum(correct).item()
