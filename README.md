# Class-Aware Adversarial Unsupervised Domain Adaptation for Linguistic Steganalysis

Official implementation of the paper:

> **Class-Aware Adversarial Unsupervised Domain Adaptation for Linguistic Steganalysis**

---

## Overview

This repository provides the code for **CAUDA**, a cross-domain linguistic steganalysis framework that adapts a steganalysis model trained on a labelled source domain to an unlabelled target domain without requiring any target-domain annotations.

The framework consists of two phases:

**Phase 1 – UDA Pre-training**
- Supervised source-domain pre-training of the feature extractor *F* and steganalysis classifier *C*
- Adversarial domain alignment via a domain discriminator *D*, optimising *L*<sub>adv</sub><sup>D</sup> (Eq. 9) and *L*<sub>adv</sub><sup>F</sup> (Eq. 10)
- Weighted Class-Aware Domain Distance (WCADD, Eq. 12) as an auxiliary metric constraint
- High-confidence pseudo-label selection on the target domain

**Phase 2 – Class-Aware Pseudo-Label Fine-tuning**
- Iterative class-balanced pseudo-label selection with confidence-based soft weighting
- Exponential growth schedule for the pseudo-label budget
- Fine-tuning of the target-domain model on the selected samples

---

## Model Architecture

```
Input Text
    │
    ▼
Frozen BERT (bert-base-uncased)
    │  token-level hidden states
    ▼
BiLSTM
    │
    ▼
Channel-wise Weighting Gate (sigmoid)
    │  domain-invariant features Ω
    ▼
┌───────────────┬──────────────────┐
│  Classifier C │  Discriminator D │
│  (cover/stego)│  (source/target) │
└───────────────┴──────────────────┘
```

---

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| torch | ≥ 1.13.1 |
| transformers | ≥ 4.30.2 |
| scikit-learn | ≥ 0.19.2 |
| tqdm | ≥ 4.66.1 |
| numpy | ≥ 1.21.6 |

A CUDA-capable GPU is recommended.

---

## Data Preparation

Each domain requires four plain-text files (one sentence per line):

```
your_data/
├── source_train_cover.txt
├── source_train_stego.txt
├── source_eval_cover.txt
├── source_eval_stego.txt
├── target_train_cover.txt
├── target_train_stego.txt
├── target_test_cover.txt
└── target_test_stego.txt
```

---

## Training

```bash
python main.py \
  --src-train-cover  your_data/source_train_cover.txt \
  --src-train-stego  your_data/source_train_stego.txt \
  --src-eval-cover   your_data/source_eval_cover.txt \
  --src-eval-stego   your_data/source_eval_stego.txt \
  --tgt-train-cover  your_data/target_train_cover.txt \
  --tgt-train-stego  your_data/target_train_stego.txt \
  --tgt-test-cover   your_data/target_test_cover.txt \
  --tgt-test-stego   your_data/target_test_stego.txt \
  --bert-path        bert-base-uncased \
  --model-root       snapshots
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--bert-path` | `bert-base-uncased` | HuggingFace model name or local BERT path |
| `--model-root` | `snapshots` | Directory for saving checkpoints |
| `--lr` | `1e-4` | Adam learning rate |
| `--batch-size` | `50` | Mini-batch size |
| `--pretrain-src-epochs` | `2` | Source pre-training epochs |
| `--pretrain-uda-epochs` | `5` | UDA adversarial pre-training epochs |
| `--pseudo-label-budget` | `20000` | Max pseudo-labelled samples per round |
| `--gpu` | `0` | GPU device index |

---

## Repository Structure

```
├── main.py          # Entry point: argument parsing and training pipeline
├── pretrain.py      # Phase 1: source pre-training + UDA adversarial training
├── Finetune.py      # Phase 2: class-aware pseudo-label fine-tuning
├── module.py        # Model definitions: FeatureExtractor, Classifier, Discriminator
├── loss.py          # WCADD loss and helper functions
├── DataLoader.py    # Dataset loading, tokenisation, and batch iteration
├── test.py          # Evaluation functions
├── utils.py         # Utilities: save_model, make_variable, LabelSmoothingCrossEntropy
└── requirements.txt
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{cauda2024,
  title   = {Class-Aware Adversarial Unsupervised Domain Adaptation for Linguistic Steganalysis},
  author  = {},
  journal = {},
  year    = {2024}
}
```
