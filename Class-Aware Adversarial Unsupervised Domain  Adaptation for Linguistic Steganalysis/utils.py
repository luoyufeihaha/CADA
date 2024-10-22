import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import linalg as la, nn

import params
from torch.autograd import Function

def save_model(args ,net, filename):
    """Save trained model."""
    # 这行代码检查模型保存的根目录是否存在，如果不存在，则创建它。 args.model_root
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    torch.save(net.state_dict(),
               os.path.join(args.model_root, filename)) # 用于构建保存路径，确保文件保存在指定的根目录下并使用指定的文件名。
    print("save pretrained model to: {}".format(os.path.join(args.model_root,
                                                             filename)))
def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs =  torch.log(x)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
