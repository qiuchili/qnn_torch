# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.complexnn import *
import dataset
import argparse
from params import Params

'''
Multi-layered Local Mixture NN
'''
class MLLM(torch.nn.Module):
    def __init__(self, opt):
        """
        max_sequence_len: input sentence length
        embedding_dim: input dimension
        num_measurements: number of measurement units, also the output dimension

        """
        super(MLLM, self).__init__()
        
        
        self.device = opt.device
        self.max_sequence_len = opt.max_sequence_length
        sentiment_lexicon = opt.sentiment_dic
        if sentiment_lexicon is not None:
            sentiment_lexicon = torch.tensor(sentiment_lexicon, dtype=torch.float)
            
        self.num_hidden_layers = len(str(opt.ngram_value).split(','))-1
        self.ngram = nn.ModuleList([NGram(gram_n = int(n_value),device = self.device) for n_value in str(opt.ngram_value).split(',') if len(n_value)>0 ])
        self.pooling_type = opt.pooling_type
        self.num_measurements = opt.measurement_size
        self.embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embedding_dim = self.embedding_matrix.shape[1]
        self.complex_embed = ComplexEmbedding(opt, self.embedding_matrix)
        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = 1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.final_mixture = ComplexMixture(use_weights= False)
        self.proj_measurements = nn.ModuleList([ComplexProjMeasurement(opt, self.embedding_dim, device = self.device) for i in range(self.num_hidden_layers)])
        self.measurement = ComplexMeasurement(self.embedding_dim, units = self.num_measurements,device = self.device)
        self.use_lexicon_as_measurement = opt.use_lexicon_as_measurement
        self.hidden_units = opt.hidden_units

        self.dense = nn.Linear(self.num_measurements,2)
        
    def forward(self, input_seq):
        """
        In the forward function we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        """
        
        amplitude_embedding, phase_embedding  = self.complex_embed(input_seq)
        weights = self.l2_norm(amplitude_embedding)
        amplitude_embedding = self.l2_normalization(amplitude_embedding)
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
        for i in range(self.num_hidden_layers):
            n_gram = self.ngram[i]
            real_n_gram_embed = n_gram(seq_embedding_real)
            imag_n_gram_embed = n_gram(seq_embedding_imag)
            n_gram_weight = n_gram(weights)
            # weights = torch.sum(n_gram_weight, dim=1)
            n_gram_weight = self.activation(n_gram_weight)
            [sentence_embedding_real, sentence_embedding_imag] = self.mixture([real_n_gram_embed, imag_n_gram_embed, n_gram_weight])
            [seq_embedding_real, seq_embedding_imag] = self.proj_measurements[i]([sentence_embedding_real, sentence_embedding_imag])

#        n_gram = self.ngram[self.num_hidden_layers]
#        n_gram_weight = n_gram(weights)
#        real_n_gram_embed = n_gram(seq_embedding_real)
#        imag_n_gram_embed = n_gram(seq_embedding_imag)
        [sentence_embedding_real, sentence_embedding_imag] = self.mixture([seq_embedding_real, seq_embedding_imag, weights])
        mea_operator = None
        if self.use_lexicon_as_measurement:
            amplitude_measure_operator, phase_measure_operator = self.complex_embed.sample(self.num_measurements)
            mea_operator = self.complex_multiply([phase_measure_operator, amplitude_measure_operator])
        output = self.measurement([sentence_embedding_real, sentence_embedding_imag], measure_operator=mea_operator)
#        output = torch.log10(output)
        output = self.dense(output)
#        output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        
        return output
