import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer
import logging
from sklearn.metrics import f1_score
import datetime


def pretrain_source_module(encoder, classifier, test_data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
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

    # print("evaluate_source_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    return acc


def pretrain_tgt_module(encoder, classifier, test_data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
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

    # print("evaluate_tgt_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    return acc


def eval_tgt(encoder,  classifier, test_data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    # for (input_data, labels) in test_data_loader:
    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        # input_dataes = make_variable(input_data, volatile=True)
        # labels = make_variable(labels
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        # preds = classifier(Filter(encoder(input_data)))
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test
    print("evaluate_source_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    # logging.info("evaluate_tgt_module Avg Loss = {}, Avg Accuracy = {:.2%}".format(loss, acc))


def eval_src(encoder, classifier, test_data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
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
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
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

    # print("evaluate_tgt_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    return acc


def eval_tgt_result(encoder, classifier, test_data_loader, args):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    cnt = 0
    len_test = 2000.0
    all_predictions = []
    all_labels = []

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    # for (input_data, labels) in test_data_loader:
    for batch in test_data_loader:
        if cnt >= len(test_data_loader):
            break
        cnt += 1

        input_data, labels = batch[0], batch[1]
        # input_dataes = make_variable(input_data, volatile=True)
        # labels = make_variable(labels)
        data_feature = encoder(input_data)
        data_feature = data_feature.reshape(data_feature.size(0), -1)

        preds = classifier(data_feature)
        # preds = classifier(Filter(encoder(input_data)))
        loss += criterion(preds, labels).detach().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

        # append predictions and labels
        all_predictions.extend(pred_cls.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = float(acc)
    loss /= len(test_data_loader)
    acc /= len_test

    # calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print("evaluate_source_module Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    logging.info("最终的运算结果evaluate_tgt_module Avg Loss = {}, Avg Accuracy = {:.2%}".format(loss, acc))

    # 获取当前时间并格式化为字符串
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 创建带有时间戳的文件名
    result_file_name = 'result_{}.txt'.format(current_time)

    # 打开文件并写入结果
    with open(os.path.join('result', result_file_name), 'a') as f:
        f.write('The testing accuracy: {:.4f} \n'.format(acc))
        f.write('The testing F1_sorce: {:.4f} \n'.format(f1))
        f.write('Source cover directory: %s \n' % args.source_cover_dir)
        f.write('target cover directory: %s \n' % args.target_cover_dir)

    return acc, f1
