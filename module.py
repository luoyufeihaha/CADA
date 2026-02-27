import os

import torch
import torch.nn as nn


class Bert(nn.Module):
    """Frozen BERT backbone for token-level feature extraction."""

    def __init__(self, args):
        super(Bert, self).__init__()
        self.bert = args.model
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x):
        context, mask = x[0], x[2]
        outputs = self.bert(context, attention_mask=mask)
        return outputs.last_hidden_state


class FeatureExtractor(nn.Module):
    """Feature extractor F: BERT → BiLSTM → channel-wise weighting gate.

    The gate applies a learned sigmoid mask over the BiLSTM output channels,
    allowing the model to suppress domain-specific and stego-irrelevant features.
    """

    def __init__(self, args, restore=None):
        super(FeatureExtractor, self).__init__()
        self.bert = Bert(args)
        self.bilstm = nn.LSTM(
            input_size=args.lstm_input_dim,
            hidden_size=args.lstm_hidden_size,
            num_layers=1,
            dropout=args.dropout,
            batch_first=True,
            bidirectional=True,
        )
        # Channel-wise weighting gate
        self.gate_fc = nn.Linear(args.lstm_hidden_size * 2, args.lstm_hidden_size * 2)
        self.gate_sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.dropout)

        self.restored = False
        if restore is not None and os.path.exists(restore):
            self.load_state_dict(torch.load(restore))
            self.restored = True
            print('FeatureExtractor restored from:', restore)

    def forward(self, x):
        bert_out = self.bert(x)
        lstm_out, _ = self.bilstm(bert_out)
        lstm_out = self.dropout(lstm_out)

        gate = self.gate_sigmoid(self.gate_fc(lstm_out))
        gate = self.dropout(gate)

        return lstm_out * gate


class Classifier(nn.Module):
    """Steganalysis classifier head: linear projection → softmax."""

    def __init__(self, args, restore=None):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(args.cls_input_size, args.num_classes)
        self.softmax = nn.Softmax(dim=-1)

        self.restored = False
        if restore is not None and os.path.exists(restore):
            self.load_state_dict(torch.load(restore))
            self.restored = True
            print('Classifier restored from:', restore)

    def forward(self, x):
        return self.softmax(self.fc(x))


class Discriminator(nn.Module):
    """Domain discriminator D: distinguishes source-domain from target-domain features."""

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(args.cls_input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        return self.layer(x)
