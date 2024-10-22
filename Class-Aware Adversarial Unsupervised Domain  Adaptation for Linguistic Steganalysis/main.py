import os
import sys
import argparse
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import logging

from DataLoader import *
import torch

from Finetune import train_tgt_finetune
from module import *
from pretrain import *

from test import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建了一个名为'LYF'的日志记录器实例，
logger = logging.getLogger('LYF')
# logging.basicConfig(level = logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    filename='Second_paper.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

seed = 20
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()

logger.info('——————————————程序开始！！！！！！——————————————')

# learning
parser.add_argument('-dropout', type=float, default=0.5,
                    help='the probability for dropout [defualt:0.5]')
parser.add_argument('-lstm_input_dim', type=float, default=768,
                    help='the size of the input feature dimension [defualt:768]')
parser.add_argument('-lstm_hidden_size', type=float, default=500,
                    help='the number of units in the hidden layer  [defualt:500]')
parser.add_argument('-cls_input_size', type=float, default=32000,
                    help='the dimension of the features received by the fully connected layer in the classifier. ['
                         'defualt:1000]')
parser.add_argument('-bottleneck_dim', type=float, default=32000,
                    help='the dimension of the elements after feature filtering  [defualt:32000]')
parser.add_argument('-num_classes', type=float, default=2,
                    help='the one domains number of categories  [defualt:2]')
parser.add_argument('-c-learning-rate', type=float, default=1e-4,
                    help='learning rate for the classifier [default: 1e-4]')
parser.add_argument('-beta1', type=float, default=0.5,
                    help='beta1 for Adam optimizer [default: 0.5]')
parser.add_argument('-beta2', type=float, default=0.9,
                    help='beta2 for Adam optimizer [default: 0.9]')

parser.add_argument('-num_epochs_pre_first', type=int, default=30,
                    help='Number of epochs for pretraining [default: 50]')
parser.add_argument('-num_epochs_pre_second', type=int, default=5,
                    help='Number of epochs for pretraining [default: 10]')

parser.add_argument('-num_epochs_finetune_all', type=int, default=10,
                    help='Number of epochs for pretraining [default: 10]')
parser.add_argument('-num_epochs_finetune_part', type=int, default=30,
                    help='Number of epochs for pretraining [default: 30]')

parser.add_argument('-batch-size', type=int, default=50,
                    help='batch size for training [default: 128]')

parser.add_argument("-src_encoder_restore", type=str, default="snapshots/LYF-source-encoder-final.pt",
                    help="path to store src_encoder files")
parser.add_argument("-src_Filter_restore", type=str, default="snapshots/LYF-src-Filter-final.pt",
                    help="path to store src_Filter files")
parser.add_argument("-src_classifier_restore", type=str, default="snapshots/LYF-source-classifier-final.pt",
                    help="path to store src_classifier files")

parser.add_argument("-tgt_encoder_restore", type=str, default="snapshots/LYF-target-encoder-final.pt",
                    help="path to store tgt_encoder files")
parser.add_argument("-tgt_Filter_restore", type=str, default="snapshots/LYF-target-Filter-final.pt",
                    help="path to store tgt_Filter files")
parser.add_argument("-tgt_classifier_restore", type=str, default="snapshots/LYF-target-classifier-final.pt",
                    help="path to store tgt_classifier files")

parser.add_argument("-model_root", type=str, default="snapshots--N->T",
                    help="path to store all files")

parser.add_argument('-EF', type=int, default=10,
                    help='Exponential Factor determines the proportion of unlabeled data selected in each training step')
parser.add_argument('-nums_u_data', type=int, default=20000,
                    help='the number of unlabeled samples [default: 20000]')

# data
parser.add_argument('-source-cover-dir', type=str,
                    default="PDTS/PLDA数据集/VLC/NEWS/1bpw/train_cover.txt",
                    help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-source-stego-dir', type=str,
                    default='PDTS/PLDA数据集/VLC/NEWS/1bpw/train_stego.txt',
                    help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-source_eval_cover_dir', type=str,
                    default='PDTS/PLDA数据集/VLC/NEWS/1bpw/test_cover.txt',
                    help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-source_eval_stego_dir', type=str,
                    default='PDTS/PLDA数据集/VLC/NEWS/1bpw/test_stego.txt',
                    help='the path of train cover data. [default:cover.txt]')

parser.add_argument('-target-cover-dir', type=str,
                    default='PDTS/PLDA数据集/VLC/TWITTER/1bpw/train_cover.txt',
                    help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-target-stego-dir', type=str,
                    default='PDTS/PLDA数据集/VLC/TWITTER/1bpw/train_stego.txt',
                    help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-test-cover-dir', type=str,
                    default='PDTS/PLDA数据集/VLC/TWITTER/1bpw/test_cover.txt',
                    help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-test-stego-dir', type=str,
                    default='PDTS/PLDA数据集/VLC/TWITTER/1bpw/test_stego.txt',
                    help='the path of train cover data. [default:cover.txt]')

# device
parser.add_argument('-no-cuda', action='store_true', default=False,
                    help='disable the gpu [default:False]')
parser.add_argument('-device', type=str, default='cuda',
                    help='device to use for trianing [default:cuda]')
parser.add_argument('-idx-gpu', type=str, default='0',
                    help='the number of gpu for training [default:0]')

args = parser.parse_args()

# 参数为Bert模型的参数
args.model = BertModel.from_pretrained('PDTS/pretrained_BERT/base_uncased')
args.tokenizer = BertTokenizer.from_pretrained('PDTS/pretrained_BERT/base_uncased')

print('\nLoading data...')

# load dataset
source_data = build_dataset(args, args.source_cover_dir, args.source_stego_dir)
source_data_eval = build_dataset(args, args.source_eval_cover_dir, args.source_eval_stego_dir)
target_data = build_dataset(args, args.target_cover_dir, args.target_stego_dir)
test_data = build_dataset(args, args.test_cover_dir, args.test_stego_dir)

source_data_loader = build_iterator(source_data, args)
source_data_loader_eval = build_iterator(source_data_eval, args)
target_data_loader = build_iterator(target_data, args)
test_data_loader = build_iterator(test_data, args)

logging.info("。。。。。。。。。。。程序开始运行。。。。。。。。。。")
logging.info("Source cover directory: %s", args.source_cover_dir)
logging.info("target cover directory: %s", args.target_cover_dir)

# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda

# load models
src_encoder = FeatureExtractor(args, args.src_encoder_restore)
src_classifier = Classifier(args, args.src_classifier_restore)

tgt_encoder = FeatureExtractor(args, args.tgt_encoder_restore)
tgt_classifier = Classifier(args, args.tgt_classifier_restore)

critic = Discriminator(args)

src_encoder_file = os.path.join(args.model_root, "LYF-source-encoder.pt")
src_classifier_file = os.path.join(args.model_root, "LYF-source-classifier.pt")

if os.path.exists(src_encoder_file):
    src_encoder.load_state_dict(torch.load(src_encoder_file))
    src_classifier.load_state_dict(torch.load(src_classifier_file))
else:
    train_pre(src_encoder, src_classifier, critic, source_data_loader, target_data_loader,
              source_data_loader_eval, test_data_loader, target_data, args)

print(">>> Pre-training completed <<<")

# 定义模型路径
encoder_file = os.path.join(args.model_root, "LYF-pretrain-target-encoder.pt")
classifier_file = os.path.join(args.model_root, "LYF-pretrain-target-classifier.pt")

# 加载权重文件到模型
tgt_encoder.load_state_dict(torch.load(encoder_file))
tgt_classifier.load_state_dict(torch.load(classifier_file))

eval_tgt(tgt_encoder, tgt_classifier, test_data_loader)

train_tgt_finetune(tgt_encoder, tgt_classifier, source_data_loader, target_data_loader, test_data_loader,
                   source_data, target_data, critic, args)

encoder_file = os.path.join(args.model_root, "LYF-best-target-encoder.pt")
classifier_file = os.path.join(args.model_root, "LYF-best-target-classifier.pt")

# 加载权重文件到模型
tgt_encoder.load_state_dict(torch.load(encoder_file))
tgt_classifier.load_state_dict(torch.load(classifier_file))

print(">>> source only <<<")
eval_src(src_encoder, src_classifier, test_data_loader)
print(">>> domain adaption <<<")
eval_tgt_result(tgt_encoder, tgt_classifier, test_data_loader, args)
