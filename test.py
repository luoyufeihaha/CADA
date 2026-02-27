import os
import logging
import datetime

import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def pretrain_source_module(encoder, classifier, test_data_loader):
    """Evaluate encoder+classifier on the source-domain test set.

    Returns accuracy (float) without printing anything.
    Used internally during Phase 1 source pre-training to track best checkpoint.
    """
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    criterion = nn.CrossEntropyLoss()

    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test

    return acc


def pretrain_tgt_module(encoder, classifier, test_data_loader):
    """Evaluate encoder+classifier on the target-domain test set.

    Returns accuracy (float) without printing anything.
    Used internally during Phase 1 UDA pre-training to track best checkpoint.
    """
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    criterion = nn.CrossEntropyLoss()

    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test

    return acc


def eval_tgt(encoder, classifier, test_data_loader):
    """Evaluate encoder+classifier on the target-domain test set and print results.

    Called after Phase 1 UDA pre-training to report the adapted model's
    initial performance before Phase 2 fine-tuning.
    """
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    criterion = nn.CrossEntropyLoss()

    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test
    print("evaluate_target_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def eval_src(encoder, classifier, test_data_loader):
    """Evaluate encoder+classifier on the source-domain test set and print results.

    Called in the final evaluation stage to report source-only baseline performance.
    """
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    criterion = nn.CrossEntropyLoss()

    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test

    print("evaluate_source_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def finetune_tgt_module(encoder, classifier, test_data_loader):
    """Evaluate encoder+classifier on the target-domain test set.

    Returns accuracy (float) without printing anything.
    Used internally during Phase 2 pseudo-label fine-tuning to track best checkpoint.
    """
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    criterion = nn.CrossEntropyLoss()

    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test

    return acc


def eval_tgt_result(encoder, classifier, test_data_loader, args):
    """Final evaluation on the target-domain test set.

    Computes accuracy and weighted F1 score, prints results, logs them,
    and writes a timestamped result file under the 'result/' directory.

    Returns:
        (acc, f1): tuple of accuracy and weighted F1 score.
    """
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0
    all_predictions = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

        all_predictions.extend(pred_cls.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test

    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print("evaluate_target_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    logging.info("evaluate_target_module Avg Loss = {}, Avg Accuracy = {:.2%}".format(loss, acc))

    # Write timestamped result file
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file_name = 'result_{}.txt'.format(current_time)

    os.makedirs('result', exist_ok=True)
    with open(os.path.join('result', result_file_name), 'a') as f:
        f.write('The testing accuracy: {:.4f} \n'.format(acc))
        f.write('The testing F1_score: {:.4f} \n'.format(f1))
        f.write('Source cover directory: %s \n' % args.source_cover_dir)
        f.write('Target cover directory: %s \n' % args.target_cover_dir)

    return acc, f1
