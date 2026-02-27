import torch
from torch import linalg as la


def wcadd(data_feature_src, data_feature_tgt, labels_src, labels_tgt, tgt_preds,
          all_weight=0.01, src_weight=0.1, tgt_weight=0.1):
    """Weighted Class-Aware Domain Distance (WCADD, Eq. 12).

    Measures the distance between source and target domains while
    incorporating class information and confidence-based sample weighting.

    Args:
        data_feature_src: source-domain feature tensor, shape (n_s, d)
        data_feature_tgt: target-domain feature tensor, shape (n_t, d)
        labels_src:       source-domain labels, shape (n_s,)
        labels_tgt:       target-domain pseudo-labels, shape (n_t,)
        tgt_preds:        target-domain class probabilities from classifier,
                          shape (n_t, num_classes)
        all_weight:       scalar weight for the cross-domain term (default: 0.01)
        src_weight:       scalar weight for the within-class source term (default: 0.1)
        tgt_weight:       scalar weight for the within-class target term (default: 0.1)

    Returns:
        Scalar WCADD loss value.
    """
    batch_size = data_feature_tgt.size(0) + data_feature_src.size(0)

    num_classes = 2
    num_samples = labels_src.size(0)

    # Source weights: inverse class frequency (proxy for one-hot uniform weighting)
    weights_src = torch.zeros(num_samples, dtype=torch.float).to(labels_src.device)
    class_counts = torch.bincount(labels_src, minlength=num_classes)
    for i in range(num_samples):
        weights_src[i] = 1.0 / class_counts[labels_src[i]]

    # Target weights: confidence-based soft weighting within each class
    num_samples = tgt_preds.size(0)
    class_sum = torch.zeros(num_classes, dtype=torch.float).to(tgt_preds.device)
    weights_tgt = torch.zeros(num_samples, dtype=torch.float).to(tgt_preds.device)

    for i in range(num_samples):
        class_sum[labels_tgt[i]] += tgt_preds[i, labels_tgt[i]]

    for i in range(num_samples):
        weights_tgt[i] = tgt_preds[i, labels_tgt[i]] / class_sum[labels_tgt[i]] \
            if class_sum[labels_tgt[i]] != 0 else 0

    # Cross-domain distance term
    loss_all = la.norm(data_feature_src - data_feature_tgt, ord=2, axis=1).sum() / float(batch_size)
    # Within-class distance terms (source and target)
    loss_src = pair_norm_with_weights(labels_src, data_feature_src, weights_src)
    loss_tgt = pair_norm_with_weights(labels_tgt, data_feature_tgt, weights_tgt)

    loss = all_weight * loss_all + src_weight * loss_src + tgt_weight * loss_tgt
    return loss


def pair_norm_with_weights(labels, features, weights):
    """Compute weighted pairwise L2 distances within each class.

    For every pair of samples (i, j) that share the same class label,
    accumulates weights[i] * weights[j] * ||features[i] - features[j]||_2
    and returns the average over all such pairs.

    Args:
        labels:   class label tensor, shape (n,)
        features: feature tensor, shape (n, d)
        weights:  per-sample weight tensor, shape (n,)

    Returns:
        Scalar weighted within-class distance (0 if no valid pair exists).
    """
    norm = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                count += 1
                norm += weights[i] * weights[j] * la.norm(features[i] - features[j], ord=2, axis=0).sum()
    return norm / count if count != 0 else 0


def entropy(input_):
    """Compute per-sample prediction entropy.

    Args:
        input_: class probability tensor, shape (n, num_classes)

    Returns:
        Per-sample entropy tensor, shape (n,)
    """
    epsilon = 1e-5
    entropy_val = -input_ * torch.log(input_ + epsilon)
    return torch.sum(entropy_val, dim=1)
