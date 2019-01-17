# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.complexnn import *
import dataset


import argparse
from params import Params

class LocalMixtureNN(torch.nn.Module):
    def __init__(self, opt):
        """
        max_sequence_len: input sentence length
        embedding_dim: input dimension
        num_measurements: number of measurement units, also the output dimension

        """
        super(LocalMixtureNN, self).__init__()
        self.max_sequence_len = opt.max_sequence_length
        self.device = opt.device
        sentiment_lexicon = opt.sentiment_dic
        if sentiment_lexicon is not None:
            sentiment_lexicon = torch.tensor(sentiment_lexicon, dtype=torch.float)
        self.ngram = nn.ModuleList([NGram(gram_n = int(n_value),device = self.device) for n_value in opt.ngram_value.split(',')])
        self.pooling_type = opt.pooling_type
        self.num_measurements = opt.measurement_size
        self.embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embedding_dim = self.embedding_matrix.shape[1]
        self.complex_embed = ComplexEmbedding(opt, self.embedding_matrix, sentiment_lexicon)
        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = 1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.measurement = ComplexMeasurement(self.embedding_dim, units = 2*self.num_measurements,device = self.device)
        self.use_lexicon_as_measurement = opt.use_lexicon_as_measurement
        self.hidden_units = opt.hidden_units
        
        self.feature_num = 0 
        for one_type in self.pooling_type.split(','):
            one_type = one_type.strip()
            if one_type == 'max':
                # max out the sequence dimension
                feature_num = 2*self.num_measurements
            elif one_type == 'average':
                # average out the sequence dimension
                feature_num = 2*self.num_measurements
            elif one_type == 'none':
                # do nothing at all, flatten
                feature_num = self.max_sequence_len*2*self.num_measurements
            elif one_type == 'max_col':
                # max out the measurement dimension
                feature_num = self.max_sequence_len
            elif one_type == 'average_col':
                # average out the measurement dimension
                feature_num = self.max_sequence_len
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                feature_num = self.max_sequence_len*2*self.num_measurements
            self.feature_num = self.feature_num + feature_num
            
        
        self.dense_1 = nn.Linear(len(self.ngram)*self.feature_num, self.hidden_units)
        self.dense_2 = nn.Linear(self.hidden_units,2)

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
        prob_list = []
        for n_gram in self.ngram:
            n_gram_embedding_real = n_gram(seq_embedding_real)
            n_gram_embedding_imag = n_gram(seq_embedding_imag)
            n_gram_weight = n_gram(weights)
            n_gram_weight = self.activation(n_gram_weight)
            
            [sentence_embedding_real, sentence_embedding_imag] = self.mixture([n_gram_embedding_real, n_gram_embedding_imag,n_gram_weight])
            
            mea_operator = None
            if self.use_lexicon_as_measurement:
                amplitude_measure_operator, phase_measure_operator = self.complex_embed.sample(self.num_measurements)
                mea_operator = self.complex_multiply([phase_measure_operator, amplitude_measure_operator])
            prob_list.append(self.measurement([sentence_embedding_real, sentence_embedding_imag], measure_operator=mea_operator))
        
        probs_tensor = torch.stack(prob_list,dim = -1)
        probs_feature = []
        for one_type in self.pooling_type.split(','):
            one_type = one_type.strip()
            if one_type == 'max':
#                probs = GlobalMaxPooling1D()(self.probs)
                # max out the sequence dimension
                probs,_ = torch.max(probs_tensor,1,False)
                
            elif one_type == 'average':
                # average out the sequence dimension
                probs = torch.mean(probs_tensor,1,False)
                
            elif one_type == 'none':
                # do nothing at all, flatten
                probs = torch.flatten(probs_tensor, start_dim=1, end_dim=2)
                
            elif one_type == 'max_col':
                # max out the measurement dimension
                probs,_ = torch.max(probs_tensor,2,False)
                
            elif one_type == 'average_col':
                # average out the measurement dimension
                probs = torch.mean(probs_tensor,2,False)
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = torch.flatten(probs_tensor, start_dim=1, end_dim=2)
            probs_feature.append(probs)
        
        probs = torch.cat(probs_feature, dim = -2)
        probs = torch.flatten(probs, start_dim = -2, end_dim = -1)

        probs = F.relu(self.dense_1(probs))
        output = self.dense_2(probs)
#        output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        
        return output