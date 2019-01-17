# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLSTM(nn.Module):
    def __init__(self, opt):
        super(TextLSTM, self).__init__()
        self.opt = opt
        embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(50, 50, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(100, 2)

    def forward(self, inp):
        text_indices = inp 
        x = self.embed(text_indices)
        output, (_, _) = self.lstm(x)
        output = torch.mean(output, dim=1)
        output = self.fc(output)
        return output