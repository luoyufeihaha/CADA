import torch
import torch.optim as optim

from DataLoader import build_iterator
from test import *
from utils import *


def train_tgt_finetune(tgt_encoder,tgt_classifier,
                       source_data_loader, target_data_loader, test_data_loader,
                       source_data, target_data, critic, args):
    optimizer_cls = optim.Adam(
        list(tgt_encoder.parameters()) + list(tgt_classifier.parameters()),
        lr=args.c_learning_rate,  # c_learning_rate = 1e-4
        betas=(args.beta1, args.beta2))  # beta1 = 0.5 beta2 = 0.9
    criterion = nn.CrossEntropyLoss()

    best_acc = finetune_tgt_module(tgt_encoder, tgt_classifier, test_data_loader)

    print("=====开始使用伪标签数据训练模型=====")
    print("使用伪标签数据前的acc：", best_acc)

    tgt_encoder.train()
    tgt_classifier.train()

    save_model(args, tgt_encoder, "LYF-best-target-encoder.pt")
    save_model(args, tgt_classifier, "LYF-best-target-classifier.pt")

    for epoch in range(10):

        tgt_encoder.load_state_dict(torch.load("LYF-best-target-encoder.pt"))
        tgt_classifier.load_state_dict(torch.load("LYF-best-target-classifier.pt"))

        acc = finetune_tgt_module(tgt_encoder, tgt_classifier, test_data_loader)
        print("加载参数后的测试：",acc)

        tgt_encoder.train()
        tgt_classifier.train()

        ###############
        ###选取伪标签###
        ###############

        tgt_features = []
        tgt_labels = []
        tgt_pseudos = []
        tgt_preds = []

        # 使用 for 循环逐行遍历列表
        for data in target_data:
            token_ids = torch.tensor(data[0])
            seq_len = torch.tensor(data[2])
            mask = torch.tensor(data[3])

            token_ids = token_ids.unsqueeze(0)
            mask = mask.unsqueeze(0)

            label = data[1]

            input_data = (token_ids, seq_len, mask)

            input_data = (input_data[0].to('cuda:0'), input_data[1].to('cuda:0'), input_data[2].to('cuda:0'))

            tgt_feature = tgt_encoder(input_data)
            tgt_feature = tgt_feature.reshape(tgt_feature.size(0), -1)
            tgt_pred = tgt_classifier(tgt_feature)

            larger_pred, pseudo_label = tgt_pred.max(1)
            larger_pred = larger_pred.item()
            pseudo_label = pseudo_label.item()

            tgt_feature = tgt_feature.cpu().detach().numpy()
            tgt_pred = tgt_pred.cpu().detach().numpy()

            tgt_features.append(tgt_feature)
            tgt_preds.append(larger_pred)
            tgt_labels.append(label)

            tgt_pseudos.append(pseudo_label)

        ratio = (epoch + 1) * args.EF / 100
        num_indices_to_select = int(args.nums_u_data * ratio)
        if num_indices_to_select >= args.nums_u_data:
            break

        # 使用类别平衡选取伪标签

        indices = sorted(range(len(tgt_preds)), key=lambda i: tgt_preds[i], reverse=True)

        selected_indices_label_1 = [indices[i] for i in range(len(indices)) if tgt_pseudos[indices[i]] == 1][
                                   :num_indices_to_select // 2]

        selected_indices_label_0 = [indices[i] for i in range(len(indices)) if tgt_pseudos[indices[i]] == 0][
                                   :num_indices_to_select // 2]

        after_indices = selected_indices_label_1 + selected_indices_label_0

        selected_labels = [tgt_labels[i] for i in after_indices]

        selected_pseudo = [tgt_pseudos[i] for i in after_indices]

        count_same = sum(1 for pseudo, label in zip(selected_pseudo, selected_labels) if pseudo == label)

        print("after_top_indices_len:", len(after_indices))
        print("after_count_same:", count_same)

        num_ones = sum(1 for i in after_indices if tgt_pseudos[i] == 1)
        num_zeros = sum(1 for i in after_indices if tgt_pseudos[i] == 0)

        print("after_num_ones:", num_ones)
        print("after_num_zeros:", num_zeros)

        train_data = []
        for i in after_indices:
            data = target_data[i]
            label = tgt_pseudos[i]
            new_data = (data[0], label, data[2], data[3])
            train_data.append(new_data)

        random.shuffle(train_data)

        train_data_loader = build_iterator(train_data, args)

        #######################
        # 使用选取的数据进行训练#
        #######################

        for step in range(20):

            # set train state for Dropout and BN layers
            tgt_encoder.train()
            tgt_classifier.train()

            for i in range(len(train_data_loader)):
                batch = next(train_data_loader)
                input_data, labels = batch[0], batch[1]
                # zero gradients for optimizer
                optimizer_cls.zero_grad()

                # compute loss for critic
                data_feature = tgt_encoder(input_data)
                data_feature = data_feature.reshape(data_feature.size(0), -1)

                preds = tgt_classifier(data_feature)

                loss_cls = criterion(preds, labels)

                # optimize source classifier
                loss_cls.backward()
                optimizer_cls.step()

            acc_tgt = finetune_tgt_module(tgt_encoder, tgt_classifier, test_data_loader)
            print("evaluate_tgt_module acc_tgt = {:4%}".format(acc_tgt))
            if acc_tgt > best_acc:
                best_acc = acc_tgt
                print("evaluate_tgt_module best_acc = {:4%}".format(best_acc))
                save_model(args, tgt_encoder, "LYF-best-target-encoder.pt")
                save_model(args, tgt_classifier, "LYF-best-target-classifier.pt")
            else:
                print("evaluate_tgt_module best_tgt = {:4%}".format(best_acc))


