import numpy as np
import torch
from torch import linalg as la
from torch.nn import functional as F

def mdd(data_feature_src, data_feature_tgt, labels_src, labels_tgt, tgt_preds, all_weight=0.01, src_weight=0.1,
        tgt_weight=0.1):

    batch_size = data_feature_tgt.size(0) + data_feature_src.size(0)

    num_classes = 2
    num_samples = labels_src.size(0)

    weights_src = torch.zeros(num_samples, dtype=torch.float).to(labels_src.device)
    class_counts = torch.bincount(labels_src, minlength=num_classes)

    for i in range(num_samples):
        weights_src[i] = 1.0 / class_counts[labels_src[i]]

    num_samples = tgt_preds.size(0)
    class_sum = torch.zeros(num_classes, dtype=torch.float).to(tgt_preds.device)
    weights_tgt = torch.zeros(num_samples, dtype=torch.float).to(tgt_preds.device)

    for i in range(num_samples):
        class_sum[labels_tgt[i]] += tgt_preds[i, labels_tgt[i]]

    for i in range(num_samples):
        weights_tgt[i] = tgt_preds[i, labels_tgt[i]] / class_sum[labels_tgt[i]] if class_sum[labels_tgt[i]] != 0 else 0

    loss_all = la.norm(data_feature_src - data_feature_tgt, ord=2, axis=1).sum() / float(batch_size)
    loss_src = pair_norm_with_weights(labels_src, data_feature_src, weights_src)
    loss_tgt = pair_norm_with_weights(labels_tgt, data_feature_tgt, weights_tgt)
    loss = all_weight * loss_all + src_weight * loss_src + tgt_weight * loss_tgt

    return loss


def pair_norm_with_weights(labels, features, weights):
    norm = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                count += 1
                norm += weights[i] * weights[j] * la.norm(features[i] - features[j], ord=2, axis=0).sum()
    return norm / count if count != 0 else 0

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy