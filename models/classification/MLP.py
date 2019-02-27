# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module): 
    def __init__(self, opt): 
        super(MLP, self).__init__() 
        self.embedding_dim = opt.embedding_size
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.Tanh()
        self.fc1 = nn.Linear(self.embedding_dim, opt.hidden_size_1)
        self.fc2 = nn.Linear(opt.hidden_size_1,opt.hidden_size_2)
        self.fc3 = nn.Linear(opt.hidden_size_2,opt.nb_classes)

    def forward(self, inp): 
        x = self.fc1(inp)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation1(x)
        x = self.fc3(x)
        output = self.activation2(x)
        return output