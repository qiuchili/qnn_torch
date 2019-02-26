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
        self.max_sequence_len = opt.max_sequence_length
        sentiment_lexicon = opt.sentiment_dic
        self.n_fold = opt.n_fold
        if sentiment_lexicon is not None:
            self.sentiment_lexicon = torch.tensor(sentiment_lexicon, dtype=torch.float).to(opt.device)
            self.vocab_size = self.sentiment_lexicon.shape[0]
            test_size = int(self.vocab_size/self.n_fold)
            perm_indices = torch.randperm(self.vocab_size)
            self.cnt = 0
            self.test_indice_list = [perm_indices[n*test_size:(n+1)*test_size] for n in range(self.n_fold)]
            self.test_mask = torch.abs(self.sentiment_lexicon)*torch.zeros(self.vocab_size, 1).scatter_(0, self.test_indice_list[self.cnt].unsqueeze(-1), 1.0).to(opt.device)
            self.train_mask = 1 - self.test_mask 
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
        self.senti_dense = nn.Linear(self.embedding_dim, 1)

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
        
        indices = torch.flatten(input_seq, -2, -1)
        if self.training:
            mask = self.train_mask.index_select(0, indices)
            senti_feat = torch.flatten(self.senti_dense(phase_embedding), 0, 1)
            senti_out = torch.tanh(senti_feat)*mask
            senti_len = torch.sum(senti_out != 0, dim=0).float() + 1 # in case of nan
            senti_tag = self.sentiment_lexicon.index_select(0, indices)  # -1, +1
            senti_loss = torch.sum(((senti_tag-senti_out)**2)*mask) / senti_len
            return senti_loss, output
        else:
            mask = self.test_mask.index_select(0, indices)
            senti_feat = torch.flatten(self.senti_dense(phase_embedding), 0, 1)
            senti_out = torch.sign(senti_feat)*mask
            senti_len = torch.sum(senti_out != 0, dim=0).float() + 1 # in case of nan
            senti_tag = self.sentiment_lexicon.index_select(0, indices) # -1, +1
            senti_acc = torch.sum((senti_out == senti_tag).float()*mask) / senti_len
            self.cnt += 1
            self.cnt %= self.n_fold
            self.test_mask = torch.abs(self.sentiment_lexicon)*torch.zeros(self.vocab_size, 1).scatter_(0, self.test_indice_list[self.cnt].unsqueeze(-1), 1.0).to(self.device)
            self.train_mask = 1 - self.test_mask
            return senti_acc, output
