# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, opt):
        super(TextCNN, self).__init__()
        self.opt = opt
        embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.channel_in = 1
        self.filter_num = 30
        self.conv11 = nn.Conv2d(self.channel_in, self.filter_num, (3, 50)) 
        self.conv12 = nn.Conv2d(self.channel_in, self.filter_num, (4, 50)) 
        self.conv13 = nn.Conv2d(self.channel_in, self.filter_num, (5, 50)) 
        self.dropout = nn.Dropout(0.5) 
        self.fc1 = nn.Linear(3*self.filter_num, 2)

    @staticmethod 
    def conv_and_pool(x, conv): 
        # x: (batch, 1, sentence_length, embed_dim) 
        x = conv(x) 
        # x: (batch, kernel_num, H_out, 1) 
        x = F.relu(x.squeeze(3)) 
        # x: (batch, kernel_num, H_out) 
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # (batch, kernel_num) 
        return x

    def forward(self, inp): 
        text_indices = inp 
        x = self.embed(text_indices) # x: (batch, sentence_length, embed_dim) 
        x = x.unsqueeze(1) # x: (batch, 1, sentence_length, embed_dim) 
        x1 = self.conv_and_pool(x, self.conv11) # (batch, filter_num) 
        x2 = self.conv_and_pool(x, self.conv12) # (batch, filter_num) 
        x3 = self.conv_and_pool(x, self.conv13) # (batch, filter_num) 
        x = torch.cat((x1, x2, x3), 1) # (batch, 3 * filter_num) 
        x = self.dropout(x) 
        output = self.fc1(x)
        return output