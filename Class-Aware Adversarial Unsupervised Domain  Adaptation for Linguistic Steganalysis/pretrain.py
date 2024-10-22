import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import params

from loss import *
from utils import *
from DataLoader import build_iterator
from test import *

logger = logging.getLogger('LYF')


def train_pre(src_encoder, src_classifier, critic, src_data_loader, tgt_data_loader, test_data_src, test_data_tgt,
              target_data, args):
    """Train classifier for source domain."""

    ##################################
    # 1. First Part of Pre-training: #
    ##################################

    optimizer_src = optim.Adam(
        list(src_encoder.parameters()) + list(src_classifier.parameters()),
        lr=args.c_learning_rate,  # c_learning_rate = 1e-4
        betas=(args.beta1, args.beta2))  # beta1 = 0.5 beta2 = 0.9
    criterion = nn.CrossEntropyLoss()
    criterionSmoothing = LabelSmoothingCrossEntropy()

    best_acc = pretrain_source_module(src_encoder, src_classifier, test_data_tgt)

    print("=====开始使用源域数据训练模型=====")
    print("使用源域数据前的acc：", best_acc)

    for epoch in range(2):

        # set train state for Dropout and BN layers
        src_encoder.train()
        src_classifier.train()

        for i in range(len(src_data_loader)):
            # for step, (input_data, labels) in enumerate(data_loader):

            # zero gradients for optimizer
            optimizer_src.zero_grad()

            batch = next(src_data_loader)
            input_data, labels = batch[0], batch[1]

            # compute loss for critic
            data_feature = src_encoder(input_data)
            data_feature = data_feature.reshape(data_feature.size(0), -1)

            preds = src_classifier(data_feature)
            loss = criterionSmoothing(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer_src.step()
            # eval model on test set

        acc = pretrain_source_module(src_encoder, src_classifier, test_data_src)

        acc_tgt = pretrain_tgt_module(src_encoder, src_classifier, test_data_tgt)

        print("evaluate_source_module Avg Accuracy = {:2%}".format(acc))
        print("evaluate_target_module Avg Accuracy = {:2%}".format(acc_tgt))

        if acc_tgt > best_acc:
            # save model parameters
            best_acc = acc_tgt
            save_model(args, src_encoder, "LYF-source-encoder.pt")
            save_model(args, src_classifier, "LYF-source-classifier.pt")
            print("保存了预训练初始化的参数模型best_acc：", best_acc)
        else:
            print("best_acc:", best_acc)

    # 定义模型路径
    encoder_file = os.path.join(args.model_root, "LYF-source-encoder.pt")
    classifier_file = os.path.join(args.model_root, "LYF-source-classifier.pt")

    # 加载权重文件到模型
    src_encoder.load_state_dict(torch.load(encoder_file))
    src_classifier.load_state_dict(torch.load(classifier_file))

    best_acc = pretrain_source_module(src_encoder, src_classifier, test_data_tgt)
    print("=====使用源域数据训练完模型后=====")
    print("使用源域数据前的acc：", best_acc)

    print("=====开始使用距离度量和域预判器训练=====")

    save_model(args, src_encoder, "LYF-pretrain-target-encoder.pt")
    save_model(args, src_classifier, "LYF-pretrain-target-classifier.pt")

    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=args.c_learning_rate,  # c_learning_rate = 1e-4
                                  betas=(args.beta1, args.beta2))  # beta1 = 0.5 beta2 = 0.9

    for epoch in range(5):
        # set train state for Dropout and BN layers
        src_encoder.train()
        src_classifier.train()
        critic.train()

        # 定义模型路径
        encoder_file = os.path.join(args.model_root, "LYF-pretrain-target-encoder.pt")
        classifier_file = os.path.join(args.model_root, "LYF-pretrain-target-classifier.pt")

        # 加载权重文件到模型
        src_encoder.load_state_dict(torch.load(encoder_file))
        src_classifier.load_state_dict(torch.load(classifier_file))

        best_acc = pretrain_source_module(src_encoder, src_classifier, test_data_tgt)

        src_encoder.train()
        src_classifier.train()

        print("此时正处于第 {} 轮循环 , 最佳的效果是 {} ".format(epoch+1, best_acc))

        ################
        ###选取伪标签####
        ################

        tgt_features = []
        tgt_labels = []
        tgt_pseudos = []
        tgt_preds = []

        for data in target_data:
            token_ids = torch.tensor(data[0])
            seq_len = torch.tensor(data[2])
            mask = torch.tensor(data[3])

            token_ids = token_ids.unsqueeze(0)
            mask = mask.unsqueeze(0)

            label = data[1]

            input_data = (token_ids, seq_len, mask)

            input_data = (input_data[0].to('cuda:0'), input_data[1].to('cuda:0'), input_data[2].to('cuda:0'))

            tgt_feature = src_encoder(input_data)
            tgt_feature = tgt_feature.reshape(tgt_feature.size(0), -1)

            tgt_pred = src_classifier(tgt_feature)

            larger_pred, pseudo_label = tgt_pred.max(1)

            larger_pred = larger_pred.item()
            pseudo_label = pseudo_label.item()

            tgt_feature = tgt_feature.cpu().detach().numpy()

            tgt_features.append(tgt_feature)
            tgt_preds.append(larger_pred)
            tgt_labels.append(label)
            tgt_pseudos.append(pseudo_label)

        num_indices_to_select = 20000

        top_indices = sorted(range(len(tgt_preds)), key=lambda i: tgt_preds[i], reverse=True)[
                      :num_indices_to_select]

        selected_labels = [tgt_labels[i] for i in top_indices]

        selected_pseudo = [tgt_pseudos[i] for i in top_indices]

        count_same = sum(1 for pseudo, label in zip(selected_pseudo, selected_labels) if pseudo == label)
        print("此次伪标签选取的数量为: ", len(top_indices))
        print("其中正确的标签数量为: ", count_same)

        train_data = []

        for i in top_indices:
            data = target_data[i]
            label = tgt_pseudos[i]
            new_data = (data[0], label, data[2], data[3])
            train_data.append(new_data)

        # 从target_data中选取对应的数据

        random.shuffle(train_data)

        train_data_loader = build_iterator(train_data, args)

        for step in range(10):

            for i in range(len(train_data_loader)):
                #################
                ###训练源域模型###
                #################

                src_encoder.train()
                src_classifier.train()
                critic.train()

                optimizer_src.zero_grad()

                batch_src = next(src_data_loader)
                input_data_src, labels_src = batch_src[0], batch_src[1]

                data_feature_src = src_encoder(input_data_src)
                data_feature_src = data_feature_src.reshape(data_feature_src.size(0), -1)

                src_preds = src_classifier(data_feature_src)

                loss_cls = criterionSmoothing(src_preds, labels_src)

                # optimize source classifier
                loss_cls.backward()
                optimizer_src.step()

                ###########################
                ########训练域判别器########
                ###########################

                input_data_src, labels_src = batch_src[0], batch_src[1]
                data_feature_src = src_encoder(input_data_src)
                data_feature_src = data_feature_src.reshape(data_feature_src.size(0), -1)

                batch_tgt = next(train_data_loader)
                input_data_tgt, labels_tgt = batch_tgt[0], batch_tgt[1]
                data_feature_tgt = src_encoder(input_data_tgt)
                data_feature_tgt = data_feature_tgt.reshape(data_feature_tgt.size(0), -1)

                optimizer_critic.zero_grad()

                feat_concat = torch.cat((data_feature_src, data_feature_tgt), 0)
                pred_concat = critic(feat_concat)

                label_src = make_variable(torch.ones(data_feature_src.size(0)).long())
                label_tgt = make_variable(torch.zeros(data_feature_tgt.size(0)).long())
                label_concat = torch.cat((label_src, label_tgt), 0)

                # compute loss for critic
                loss_critic = criterion(pred_concat, label_concat)
                loss_critic.backward()

                # optimize critic
                optimizer_critic.step()

                ###########################
                ###使用度量距离和域判别器#####
                ###########################

                input_data_src, labels_src = batch_src[0], batch_src[1]
                data_feature_src = src_encoder(input_data_src)
                data_feature_src = data_feature_src.reshape(data_feature_src.size(0), -1)
                src_preds = src_classifier(data_feature_src)

                input_data_tgt, labels_tgt = batch_tgt[0], batch_tgt[1]
                data_feature_tgt = src_encoder(input_data_tgt)
                data_feature_tgt = data_feature_tgt.reshape(data_feature_tgt.size(0), -1)
                tgt_preds = src_classifier(data_feature_tgt)

                # zero gradients for optimizer
                optimizer_src.zero_grad()

                loss_mdd = mdd(data_feature_src, data_feature_tgt, labels_src, labels_tgt, tgt_preds)

                # predict on discriminator
                pred_cri = critic(data_feature_tgt)

                # prepare fake labels
                label_tgt = make_variable(torch.ones(data_feature_tgt.size(0)).long())

                # 计算每个样本的熵值
                entropy = Entropy(tgt_preds)

                entropy = 1.0 + torch.exp(-entropy)

                weight = entropy / torch.sum(entropy).detach().item()

                loss_cls = torch.sum(weight.view(-1, 1) * criterion(pred_cri, label_tgt)) / torch.sum(
                    weight).detach().item()

                if i == 0:
                    print("loss:",loss_cls)
                    print("loss_mdd:",loss_mdd)

                loss = loss_cls + 5 * loss_mdd
                loss.backward()
                optimizer_src.step()

            acc = pretrain_source_module(src_encoder, src_classifier, test_data_src)

            acc_tgt = pretrain_tgt_module(src_encoder, src_classifier, test_data_tgt)

            print("evaluate_source_module Avg Accuracy = {:2%}".format(acc))
            print("evaluate_target_module Avg Accuracy = {:2%}".format(acc_tgt))

            if acc_tgt > best_acc:
                # save model parameters
                best_acc = acc_tgt
                save_model(args, src_encoder, "LYF-pretrain-target-encoder.pt")
                save_model(args, src_classifier, "LYF-pretrain-target-classifier.pt")
                print("保存了预训练的参数模型best_acc：", best_acc)
            else:
                print("best_acc:", best_acc)

