# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.complexnn import *
import dataset
import argparse
from params import Params

'''
Quantum Context-Preserving-Tranformation Netowrk
'''
class QTNET(torch.nn.Module):
    def __init__(self, opt):
        super(QTNET, self).__init__()
        self.device = opt.device
        self.ngram = NGram(gram_n = 3, device = self.device)
        self.num_measurements = opt.measurement_size
        self.embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.sentiment_lexicon = torch.tensor(opt.sentiment_dic, dtype=torch.float).to(opt.device)
        self.vocab_size = self.embedding_matrix.shape[0]
        self.embedding_dim = self.embedding_matrix.shape[1]
        self.complex_embed = ComplexEmbedding(opt, self.embedding_matrix)
        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = 1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.proj_measurements = nn.ModuleList([ComplexProjMeasurement(opt, self.embedding_dim, device = self.device) for i in range(opt.num_hidden_layers)])
        self.measurement = ComplexMeasurement(self.embedding_dim, units = self.num_measurements,device = self.device)
        self.num_hidden_layers = opt.num_hidden_layers
        self.dense = nn.Linear(self.embedding_dim, 2)
        self.senti_dense = nn.Linear(self.embedding_dim, 2)

    def forward(self, input_seq):
        """
        In the forward function we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        """
        
        amplitude_embedding, phase_embedding  = self.complex_embed(input_seq)
        weights = self.l2_norm(amplitude_embedding)
        amplitude_embedding = self.l2_normalization(amplitude_embedding)
        n_gram_weight = self.ngram(weights)
        n_gram_weight = self.activation(n_gram_weight)
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
        
        for i in range(self.num_hidden_layers):
            real_n_gram_embed = self.ngram(seq_embedding_real)
            imag_n_gram_embed = self.ngram(seq_embedding_imag)
            [sentence_embedding_real, sentence_embedding_imag] = self.mixture([real_n_gram_embed, imag_n_gram_embed, n_gram_weight])
            [seq_embedding_real, seq_embedding_imag] = self.proj_measurements[i]([sentence_embedding_real, sentence_embedding_imag])
        
        real_n_gram_embed = self.ngram(seq_embedding_real)
        imag_n_gram_embed = self.ngram(seq_embedding_imag)
        [sentence_embedding_real, sentence_embedding_imag] = self.mixture([real_n_gram_embed, imag_n_gram_embed, n_gram_weight])
        feats = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        flatten_feats = torch.flatten(self.senti_dense(feats), 0, 1)
        senti_out = torch.tanh(flatten_feats)
        indices = torch.flatten(input_seq, -2, -1)
        senti_tag = (self.sentiment_lexicon.index_select(0, indices).squeeze(-1).long() + 1) / 2
        max_feats = torch.max(feats, dim=1)[0]
        output = self.dense(max_feats)
        return senti_out, senti_tag, output
