import os
import random
import logging
import argparse

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from DataLoader import build_dataset, build_iterator
from Finetune import train_tgt_finetune
from module import FeatureExtractor, Classifier, Discriminator
from pretrain import train_pre
from test import eval_src, eval_tgt, eval_tgt_result

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    filename='cada.log',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('CADA')

# ── Checkpoint filenames ──────────────────────────────────────────────────────
# Phase 1: source-domain supervised pre-training
CKPT_SRC_ENCODER     = 'source_encoder.pt'
CKPT_SRC_CLASSIFIER  = 'source_classifier.pt'
# Phase 1: after UDA adversarial pre-training
CKPT_UDA_ENCODER     = 'uda_encoder.pt'
CKPT_UDA_CLASSIFIER  = 'uda_classifier.pt'
# Phase 2: best model after class-aware pseudo-label fine-tuning
CKPT_BEST_ENCODER    = 'best_encoder.pt'
CKPT_BEST_CLASSIFIER = 'best_classifier.pt'


def build_parser():
    parser = argparse.ArgumentParser(
        description='Class-Aware Adversarial Unsupervised Domain Adaptation '
                    'for Linguistic Steganalysis (CADA)'
    )

    # ── Model architecture ────────────────────────────────────────────────────
    arch = parser.add_argument_group('Model Architecture')
    arch.add_argument('--dropout', type=float, default=0.5,
                      help='dropout probability (default: 0.5)')
    arch.add_argument('--lstm-input-dim', type=int, default=768,
                      dest='lstm_input_dim',
                      help='BERT output size fed into BiLSTM (default: 768)')
    arch.add_argument('--lstm-hidden-size', type=int, default=500,
                      dest='lstm_hidden_size',
                      help='BiLSTM hidden size per direction (default: 500)')
    arch.add_argument('--cls-input-size', type=int, default=32000,
                      dest='cls_input_size',
                      help='flattened feature size entering the classifier head '
                           '(seq_len × lstm_hidden_size × 2, default: 32000)')
    arch.add_argument('--bottleneck-dim', type=int, default=32000,
                      dest='bottleneck_dim',
                      help='bottleneck feature dimension (default: 32000)')
    arch.add_argument('--num-classes', type=int, default=2,
                      dest='num_classes',
                      help='number of steganalysis categories (default: 2)')

    # ── Optimisation ──────────────────────────────────────────────────────────
    optim = parser.add_argument_group('Optimisation')
    optim.add_argument('--lr', type=float, default=1e-4,
                       dest='c_learning_rate',
                       help='learning rate for Adam (default: 1e-4)')
    optim.add_argument('--beta1', type=float, default=0.5,
                       help='Adam beta1 (default: 0.5)')
    optim.add_argument('--beta2', type=float, default=0.9,
                       help='Adam beta2 (default: 0.9)')
    optim.add_argument('--batch-size', type=int, default=50,
                       dest='batch_size',
                       help='mini-batch size (default: 50)')

    # ── Training schedule ─────────────────────────────────────────────────────
    sched = parser.add_argument_group('Training Schedule')
    sched.add_argument('--pretrain-src-epochs', type=int, default=2,
                       dest='num_epochs_pre_first',
                       help='epochs for supervised source pre-training (default: 2)')
    sched.add_argument('--pretrain-uda-epochs', type=int, default=5,
                       dest='num_epochs_pre_second',
                       help='outer epochs for UDA adversarial pre-training (default: 5)')
    sched.add_argument('--finetune-all-epochs', type=int, default=10,
                       dest='num_epochs_finetune_all',
                       help='epochs for full-network fine-tuning (default: 10)')
    sched.add_argument('--finetune-part-epochs', type=int, default=30,
                       dest='num_epochs_finetune_part',
                       help='epochs for partial fine-tuning (default: 30)')
    sched.add_argument('--pseudo-label-budget', type=int, default=20000,
                       dest='nums_u_data',
                       help='max pseudo-labelled target samples per round (default: 20000)')
    sched.add_argument('--pseudo-label-factor', type=int, default=10,
                       dest='EF',
                       help='growth factor for pseudo-label selection (default: 10)')

    # ── Data paths ────────────────────────────────────────────────────────────
    data = parser.add_argument_group('Data Paths')
    data.add_argument('--src-train-cover', type=str, required=True,
                      dest='source_cover_dir',
                      help='source-domain training cover texts')
    data.add_argument('--src-train-stego', type=str, required=True,
                      dest='source_stego_dir',
                      help='source-domain training stego texts')
    data.add_argument('--src-eval-cover', type=str, required=True,
                      dest='source_eval_cover_dir',
                      help='source-domain evaluation cover texts')
    data.add_argument('--src-eval-stego', type=str, required=True,
                      dest='source_eval_stego_dir',
                      help='source-domain evaluation stego texts')
    data.add_argument('--tgt-train-cover', type=str, required=True,
                      dest='target_cover_dir',
                      help='target-domain (unlabelled) training cover texts')
    data.add_argument('--tgt-train-stego', type=str, required=True,
                      dest='target_stego_dir',
                      help='target-domain (unlabelled) training stego texts')
    data.add_argument('--tgt-test-cover', type=str, required=True,
                      dest='test_cover_dir',
                      help='target-domain test cover texts')
    data.add_argument('--tgt-test-stego', type=str, required=True,
                      dest='test_stego_dir',
                      help='target-domain test stego texts')

    # ── Paths ─────────────────────────────────────────────────────────────────
    paths = parser.add_argument_group('Paths')
    paths.add_argument('--bert-path', type=str, default='bert-base-uncased',
                       dest='bert_path',
                       help='HuggingFace model name or local path for BERT '
                            '(default: bert-base-uncased)')
    paths.add_argument('--model-root', type=str, default='snapshots',
                       dest='model_root',
                       help='directory for saving/loading checkpoints (default: snapshots)')

    # ── Device ────────────────────────────────────────────────────────────────
    dev = parser.add_argument_group('Device')
    dev.add_argument('--gpu', type=str, default='0',
                     dest='idx_gpu',
                     help='GPU device index (default: 0)')
    dev.add_argument('--no-cuda', action='store_true', default=False,
                     help='disable GPU training')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Device setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu
    args.device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'

    os.makedirs(args.model_root, exist_ok=True)

    logger.info('Experiment started')
    logger.info('Source domain: %s', args.source_cover_dir)
    logger.info('Target domain: %s', args.target_cover_dir)

    # Load pre-trained BERT tokenizer and backbone (weights frozen in Bert module)
    args.model = BertModel.from_pretrained(args.bert_path)
    args.tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # ── Data loading ──────────────────────────────────────────────────────────
    print('Loading datasets ...')
    source_data      = build_dataset(args, args.source_cover_dir, args.source_stego_dir)
    source_data_eval = build_dataset(args, args.source_eval_cover_dir, args.source_eval_stego_dir)
    target_data      = build_dataset(args, args.target_cover_dir, args.target_stego_dir)
    test_data        = build_dataset(args, args.test_cover_dir, args.test_stego_dir)

    source_data_loader      = build_iterator(source_data, args)
    source_data_loader_eval = build_iterator(source_data_eval, args)
    target_data_loader      = build_iterator(target_data, args)
    test_data_loader        = build_iterator(test_data, args)

    # ── Model initialisation ──────────────────────────────────────────────────
    src_encoder    = FeatureExtractor(args)
    src_classifier = Classifier(args)
    tgt_encoder    = FeatureExtractor(args)
    tgt_classifier = Classifier(args)
    critic         = Discriminator(args)

    # ── Phase 1: UDA pre-training ─────────────────────────────────────────────
    # Skip pre-training if cached checkpoints already exist.
    src_encoder_ckpt    = os.path.join(args.model_root, CKPT_SRC_ENCODER)
    src_classifier_ckpt = os.path.join(args.model_root, CKPT_SRC_CLASSIFIER)

    if os.path.exists(src_encoder_ckpt):
        src_encoder.load_state_dict(torch.load(src_encoder_ckpt))
        src_classifier.load_state_dict(torch.load(src_classifier_ckpt))
        print('Loading cached source checkpoints; skipping pre-training ...')
    else:
        train_pre(src_encoder, src_classifier, critic,
                  source_data_loader, target_data_loader,
                  source_data_loader_eval, test_data_loader,
                  target_data, args)

    print('>>> Phase 1: pre-training completed <<<')

    # Load UDA-adapted weights into the target-domain model
    print('Loading UDA-adapted checkpoints into target model ...')
    tgt_encoder.load_state_dict(
        torch.load(os.path.join(args.model_root, CKPT_UDA_ENCODER)))
    tgt_classifier.load_state_dict(
        torch.load(os.path.join(args.model_root, CKPT_UDA_CLASSIFIER)))
    print('UDA-adapted checkpoints loaded successfully.')

    eval_tgt(tgt_encoder, tgt_classifier, test_data_loader)

    # ── Phase 2: Class-aware pseudo-label fine-tuning ─────────────────────────
    train_tgt_finetune(tgt_encoder, tgt_classifier,
                       source_data_loader, target_data_loader,
                       test_data_loader, source_data, target_data,
                       critic, args)

    print('>>> Phase 2: fine-tuning completed <<<')

    # ── Final evaluation ──────────────────────────────────────────────────────
    print('Loading best fine-tuned checkpoints for final evaluation ...')
    tgt_encoder.load_state_dict(
        torch.load(os.path.join(args.model_root, CKPT_BEST_ENCODER)))
    tgt_classifier.load_state_dict(
        torch.load(os.path.join(args.model_root, CKPT_BEST_CLASSIFIER)))
    print('Best checkpoints loaded successfully.')

    print('>>> Source-only performance <<<')
    eval_src(src_encoder, src_classifier, test_data_loader)
    print('>>> After domain adaptation <<<')
    eval_tgt_result(tgt_encoder, tgt_classifier, test_data_loader, args)


if __name__ == '__main__':
    main()
