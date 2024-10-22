import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import math
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        self.bert = args.model
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x):
        context, mask = x[0], x[2]
        outputs = self.bert(context, attention_mask=mask)
        encoder_outputs, text_cls = outputs.last_hidden_state, outputs.pooler_output

        return encoder_outputs


class FeatureExtractor(nn.Module):

    def __init__(self, args, restore=None):
        super(FeatureExtractor, self).__init__()
        self.bert = Bert(args)
        self.bilstm = nn.LSTM(input_size=args.lstm_input_dim,  # 768
                              hidden_size=args.lstm_hidden_size,  # 500
                              num_layers=1,
                              dropout=args.dropout,
                              batch_first=True,
                              bidirectional=True)

        self.dropout = nn.Dropout(p=0.5)

        self.restored = False

        # Restore model weights if specified
        if restore is not None and os.path.exists(restore):
            self.load_state_dict(torch.load(restore))
            self.restored = True
            print("Model restored from:", restore)

    # self.feature_filter = FeatureFilter(args)

    def forward(self, data):  # computes activations for BOTH domains
        bert_features = self.bert(data)
        lstm_features, _ = self.bilstm(bert_features)
        features = self.dropout(lstm_features)
        filtered_features = self.fc(features)
        features_weight = self.sigmoid(filtered_features)

        features_weight = self.dropout(features_weight)

        return features * features_weight


class FeatureFilter(nn.Module):
    def __init__(self, args, restore=None):
        super(FeatureFilter, self).__init__()
        self.fc = nn.Linear(args.lstm_hidden_size * 2, args.lstm_hidden_size * 2)  # Adjust output size as needed
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout)

        self.restored = False  # 初始化 restored 属性为 False

        # Restore model weights if specified
        if restore is not None and os.path.exists(restore):
            self.load_state_dict(torch.load(restore))
            self.restored = True
            print("Model restored from:", restore)

    def forward(self, features):
        filtered_features = self.fc(features)
        features_weight = self.sigmoid(filtered_features)
        return features * features_weight


class Classifier(nn.Module):
    def __init__(self, args, restore=None):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(args.cls_input_size, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.restored = False

        # Restore model weights if specified
        if restore is not None and os.path.exists(restore):
            self.load_state_dict(torch.load(restore))
            self.restored = True
            print("Model restored from:", restore)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, args):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(args.cls_input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
