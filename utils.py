import os

import torch
from torch import nn
from torch.autograd import Variable


def save_model(args, net, filename):
    """Save model state dict to args.model_root/filename."""
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    torch.save(net.state_dict(),
               os.path.join(args.model_root, filename))
    print("Saved model to: {}".format(os.path.join(args.model_root, filename)))


def make_variable(tensor, volatile=False):
    """Move tensor to GPU (if available) and wrap in a Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Mixes the hard target distribution with a uniform distribution
    to prevent overconfident predictions.
    """

    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = torch.log(x)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
