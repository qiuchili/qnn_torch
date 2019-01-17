# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.complexnn import *

class ComplexFastText(nn.Module): 
    def __init__(self, opt): 
        super(ComplexFastText, self).__init__() 
        self.opt = opt
        embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.complex_embed = ComplexEmbedding(opt, embedding_matrix)
        self.l2_normalization = L2Normalization(dim=-1)
        self.multiply = ComplexMultiply()
        self.linear = nn.Linear(100, 400)
        self.bn = nn.BatchNorm1d(400)
        self.fc = nn.Linear(400, 2) 

    def forward(self, inp): 
        text_indices = inp
        amplitude_embed, phase_embed = self.complex_embed(text_indices)
        amplitude_embed = self.l2_normalization(amplitude_embed)
        real_embed, imag_embed = self.multiply([phase_embed, amplitude_embed])
        embed = torch.cat([real_embed, imag_embed], dim=2)
        output = torch.mean(embed, dim=1)
        output = self.linear(output)
        output = self.bn(output) 
        output = self.fc(output)
        return output