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
