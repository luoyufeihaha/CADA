# Class-Aware Adversarial Unsupervised Domain Adaptation for Linguistic Steganalysis (CADA)

Official implementation of:

> **Zhen Yang, Yufei Luo, Jinshuai Yang, Xin Xu, Ru Zhang*, Yongfeng Huang**  
> *Class-aware Adversarial Unsupervised Domain Adaptation for Linguistic Steganalysis*  
> IEEE Transactions on Information Forensics and Security (TIFS), 2025  
> DOI: https://doi.org/10.1109/TIFS.2025.3569409  

---

## Architecture

![CADA Architecture](images/The%20architecture%20of%20the%20proposed%20linguistic%20steganalysis%20method%20CADA.png)

## Main Results

![Comparison of Domain Adaptation Steganalysis Methods](images/Comparison%20of%20Domain%20Adaptation%20Steganalysis%20Methods.png)

## Overview

Cross-domain linguistic steganalysis aims to detect stego texts in an unlabeled target domain using a model trained on a labeled source domain.  

Existing methods mainly reduce marginal distribution discrepancy between domains, but often suffer from:

- **Class-misalignment**: incorrect alignment between cover and stego texts across domains  
- **Class-indistinction**: insufficient class separation in the target domain  

To address these issues, we propose **CADA**, a two-stage class-aware adversarial domain adaptation framework.

---

## Framework

CADA consists of two stages:

### Phase 1 – Class-aware Adversarial Pre-Training (CAPT)

- Supervised source-domain training
- Adversarial domain alignment (feature extractor vs. domain discriminator)
- **Weighted Class-Aware Domain Distance (WCADD)**
- Class-Aware Label Smoothing (CALS)

### Phase 2 – Class-aware Fine-Tuning (CFT)

- Iterative pseudo-label selection
- Confidence-aware soft weighting
- Class-balanced sampling
- Progressive pseudo-label growth

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


---

## Dataset

We use publicly available datasets introduced in:

Wen et al.,  
**SCL-Stega: Exploring Advanced Objective in Linguistic Steganalysis using Contrastive Learning**  
IH&MMSec 2023  

```bibtex
@inproceedings{wen2023scl,
  title={Scl-stega: Exploring advanced objective in linguistic steganalysis using contrastive learning},
  author={Wen, Juan and Gao, Liting and Fan, Guangying and Zhang, Ziwei and Jia, Jianghao and Xue, Yiming},
  booktitle={Proceedings of the 2023 ACM Workshop on Information Hiding and Multimedia Security},
  pages={97--102},
  year={2023}
}
```

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
@article{yang2025class,
  title={Class-aware adversarial unsupervised domain adaptation for linguistic steganalysis},
  author={Yang, Zhen and Luo, Yufei and Yang, Jinshuai and Xu, Xin and Zhang, Ru and Huang, Yongfeng},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
```

## License

This repository is released under the MIT License.

## Contact

If you have any questions about the paper, the implementation, or encounter issues when reproducing the results, please feel free to reach out:

- **Yufei Luo**: luoyf@bupt.edu.cn  

We warmly welcome discussions, suggestions, and potential collaborations on related topics, including linguistic steganalysis, domain adaptation, and adversarial learning.
