import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import random

from loss import wcadd
from utils import *
from DataLoader import build_iterator
from test import *

logger = logging.getLogger('CADA')


def train_pre(src_encoder, src_classifier, critic, src_data_loader, tgt_data_loader, test_data_src, test_data_tgt,
              target_data, args):
    """UDA pre-training pipeline.

    Phase 1: supervised source-domain pre-training of F and C.
    Phase 2: adversarial UDA training with MDD distance constraint and
             pseudo-label selection on the target domain.
    """

    ##################################
    # Phase 1: Source Pre-training   #
    ##################################

    optimizer_src = optim.Adam(
        list(src_encoder.parameters()) + list(src_classifier.parameters()),
        lr=args.c_learning_rate,
        betas=(args.beta1, args.beta2))
    criterion = nn.CrossEntropyLoss()
    criterionSmoothing = LabelSmoothingCrossEntropy()

    best_acc = pretrain_source_module(src_encoder, src_classifier, test_data_tgt)

    print("===== Phase 1: source-domain pre-training started =====")
    print("Initial target-domain accuracy: ", best_acc)

    for epoch in range(2):

        # set train state for Dropout and BN layers
        src_encoder.train()
        src_classifier.train()

        for i in range(len(src_data_loader)):

            optimizer_src.zero_grad()

            batch = next(src_data_loader)
            input_data, labels = batch[0], batch[1]

            data_feature = src_encoder(input_data)
            data_feature = data_feature.reshape(data_feature.size(0), -1)

            preds = src_classifier(data_feature)
            loss = criterionSmoothing(preds, labels)

            loss.backward()
            optimizer_src.step()

        acc = pretrain_source_module(src_encoder, src_classifier, test_data_src)
        acc_tgt = pretrain_tgt_module(src_encoder, src_classifier, test_data_tgt)

        print("Source Avg Accuracy = {:2%}".format(acc))
        print("Target Avg Accuracy = {:2%}".format(acc_tgt))

        if acc_tgt > best_acc:
            best_acc = acc_tgt
            save_model(args, src_encoder, "source_encoder.pt")
            save_model(args, src_classifier, "source_classifier.pt")
            print("Source pre-training checkpoint saved. best_acc: ", best_acc)
        else:
            print("best_acc:", best_acc)

    # Load best source checkpoint
    encoder_file = os.path.join(args.model_root, "source_encoder.pt")
    classifier_file = os.path.join(args.model_root, "source_classifier.pt")

    src_encoder.load_state_dict(torch.load(encoder_file))
    src_classifier.load_state_dict(torch.load(classifier_file))

    best_acc = pretrain_source_module(src_encoder, src_classifier, test_data_tgt)
    print("===== Phase 1: source-domain pre-training completed =====")
    print("Target-domain accuracy after source pre-training: ", best_acc)

    print("===== Phase 2: UDA adversarial pre-training started =====")

    save_model(args, src_encoder, "uda_encoder.pt")
    save_model(args, src_classifier, "uda_classifier.pt")

    optimizer_critic = optim.Adam(
        critic.parameters(),
        lr=args.c_learning_rate,
        betas=(args.beta1, args.beta2))

    for epoch in range(5):
        # set train state for Dropout and BN layers
        src_encoder.train()
        src_classifier.train()
        critic.train()

        # Load best UDA checkpoint at the start of each epoch
        encoder_file = os.path.join(args.model_root, "uda_encoder.pt")
        classifier_file = os.path.join(args.model_root, "uda_classifier.pt")

        src_encoder.load_state_dict(torch.load(encoder_file))
        src_classifier.load_state_dict(torch.load(classifier_file))

        best_acc = pretrain_source_module(src_encoder, src_classifier, test_data_tgt)

        src_encoder.train()
        src_classifier.train()

        print("Epoch {} / 5 -- current best_acc: {}".format(epoch + 1, best_acc))

        ######################################
        # Pseudo-label selection on target   #
        ######################################

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
        print("Pseudo-labels selected: ", len(top_indices))
        print("Correct pseudo-labels:  ", count_same)

        train_data = []

        for i in top_indices:
            data = target_data[i]
            label = tgt_pseudos[i]
            new_data = (data[0], label, data[2], data[3])
            train_data.append(new_data)

        random.shuffle(train_data)

        train_data_loader = build_iterator(train_data, args)

        for step in range(10):

            for i in range(len(train_data_loader)):

                ######################################
                # Step 1: Source classification loss #
                ######################################

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

                loss_cls.backward()
                optimizer_src.step()

                ######################################
                # Step 2: Train discriminator D      #
                # Objective: L_adv_D (Eq. 9)         #
                ######################################

                input_data_src, labels_src = batch_src[0], batch_src[1]
                data_feature_src = src_encoder(input_data_src)
                data_feature_src = data_feature_src.reshape(data_feature_src.size(0), -1)

                batch_tgt = next(train_data_loader)
                input_data_tgt, labels_tgt = batch_tgt[0], batch_tgt[1]
                data_feature_tgt = src_encoder(input_data_tgt)
                data_feature_tgt = data_feature_tgt.reshape(data_feature_tgt.size(0), -1)

                optimizer_critic.zero_grad()

                feat_concat = torch.cat((data_feature_src.detach(), data_feature_tgt.detach()), 0)
                pred_concat = critic(feat_concat)

                label_src = make_variable(torch.ones(data_feature_src.size(0)).long())
                label_tgt = make_variable(torch.zeros(data_feature_tgt.size(0)).long())
                label_concat = torch.cat((label_src, label_tgt), 0)

                loss_adv_D = criterion(pred_concat, label_concat)
                loss_adv_D.backward()

                optimizer_critic.step()

                ######################################
                # Step 3: Train feature extractor F  #
                # Objective: L_adv_F + MDD (Eq. 10) #
                ######################################

                input_data_src, labels_src = batch_src[0], batch_src[1]
                data_feature_src = src_encoder(input_data_src)
                data_feature_src = data_feature_src.reshape(data_feature_src.size(0), -1)
                src_preds = src_classifier(data_feature_src)

                input_data_tgt, labels_tgt = batch_tgt[0], batch_tgt[1]
                data_feature_tgt = src_encoder(input_data_tgt)
                data_feature_tgt = data_feature_tgt.reshape(data_feature_tgt.size(0), -1)
                tgt_preds = src_classifier(data_feature_tgt)

                optimizer_src.zero_grad()

                loss_wcadd = wcadd(data_feature_src, data_feature_tgt, labels_src, labels_tgt, tgt_preds)

                # L_adv_F: labels are the opposite of L_adv_D to fool D.
                # No detach() here — gradients must flow back to F (src_encoder).
                feat_concat_adv = torch.cat((data_feature_src, data_feature_tgt), 0)
                pred_concat_adv = critic(feat_concat_adv)

                label_src_fake = make_variable(torch.zeros(data_feature_src.size(0)).long())
                label_tgt_fake = make_variable(torch.ones(data_feature_tgt.size(0)).long())
                label_concat_adv = torch.cat((label_src_fake, label_tgt_fake), 0)

                loss_adv_F = criterion(pred_concat_adv, label_concat_adv)

                if i == 0:
                    print("L_adv_F:  ", loss_adv_F)
                    print("L_wcadd:  ", loss_wcadd)

                loss = loss_adv_F + 5 * loss_wcadd
                loss.backward()
                optimizer_src.step()

            acc = pretrain_source_module(src_encoder, src_classifier, test_data_src)
            acc_tgt = pretrain_tgt_module(src_encoder, src_classifier, test_data_tgt)

            print("Source Avg Accuracy = {:2%}".format(acc))
            print("Target Avg Accuracy = {:2%}".format(acc_tgt))

            if acc_tgt > best_acc:
                best_acc = acc_tgt
                save_model(args, src_encoder, "uda_encoder.pt")
                save_model(args, src_classifier, "uda_classifier.pt")
                print("UDA checkpoint saved. best_acc: ", best_acc)
            else:
                print("best_acc:", best_acc)
