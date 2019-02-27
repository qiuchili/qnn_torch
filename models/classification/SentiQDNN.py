# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.complexnn import *
import dataset
import argparse
from params import Params

class SentiQDNN(torch.nn.Module):
    def __init__(self, opt):
        super(SentiQDNN, self).__init__()
        self.device = opt.device
        self.variant = opt.variant
        sentiment_lexicon = opt.sentiment_dic
        if sentiment_lexicon is not None:
            self.sentiment_lexicon = torch.tensor(sentiment_lexicon, dtype=torch.float).to(opt.device)
        self.num_measurements = opt.measurement_size
        self.embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embedding_dim = self.embedding_matrix.shape[1]
        self.complex_embed = ComplexEmbedding(opt, self.embedding_matrix, self.sentiment_lexicon)
        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = 1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.measurement = ComplexMeasurement(self.embedding_dim, units = self.num_measurements,device = self.device)
        self.dense = nn.Linear(self.num_measurements, 2)
        self.senti_dense = nn.Linear(self.embedding_dim, 18)

    def forward(self, input_seq):
        """
        In the forward function we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        """
        
        amplitude_embedding, phase_embedding  = self.complex_embed(input_seq)
        weights = self.l2_norm(amplitude_embedding)
        amplitude_embedding = self.l2_normalization(amplitude_embedding)
        weights = self.activation(weights)
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
        [sentence_embedding_real, sentence_embedding_imag] = self.mixture([seq_embedding_real, seq_embedding_imag,weights])
        
        output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        output = self.dense(output)
        
        indices = input_seq.flatten(0, 1)
        if self.training:
            senti_out = self.senti_dense(phase_embedding).flatten(0, 1)
            senti_tag = self.sentiment_lexicon.index_select(0, indices).squeeze(-1).long()
            return senti_out, senti_tag, output
        else:
            senti_out = torch.argmax(self.senti_dense(phase_embedding).flatten(0, 1), dim=-1)
            senti_tag = self.sentiment_lexicon.index_select(0, indices).long()
            senti_acc = torch.sum((senti_out == senti_tag).float()) / senti_out.size(0)
            return senti_acc, output
