import os
import pandas as pd
import numpy as np
import json
import pickle
import math

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, embed_dim, num_class):
        super(MLP, self).__init__()
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, emb):
        output = self.fc(emb)
        output = F.log_softmax(output, dim=1)
        return output


class NN(nn.Module):
    def __init__(self, emb_dim, out_dim, dropout=0.25, n_hid1=256, n_hid2=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, n_hid1),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid1),
            nn.Dropout(dropout),
            nn.Linear(n_hid1, n_hid2 // 4),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid2 // 4),
            nn.Dropout(dropout),
            nn.Linear(n_hid2 // 4, out_dim),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
