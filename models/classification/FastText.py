# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module): 
    def __init__(self, opt): 
        super(FastText, self).__init__() 
        self.opt = opt
        embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embed = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.linear = nn.Linear(50, 200)
        self.bn = nn.BatchNorm1d(200)
        self.fc = nn.Linear(200, 2) 

    def forward(self, inp):
        text_indices = inp 
        x = self.embed(text_indices) 
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        x = self.bn(x) 
        output = self.fc(x)
        return output