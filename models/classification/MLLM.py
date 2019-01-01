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
        self.max_sequence_len = opt.max_sequence_length
        sentiment_lexicon = opt.sentiment_dic
        if sentiment_lexicon is not None:
            sentiment_lexicon = torch.tensor(sentiment_lexicon, dtype=torch.float)
        self.ngram = [NGram(gram_n = int(n_value)) for n_value in opt.ngram_value.split(',')]
        self.pooling_type = opt.pooling_type
        self.num_measurements = opt.measurement_size
        self.embedding_matrix = torch.tensor(opt.lookup_table)
        self.embedding_dim = self.embedding_matrix.shape[1]
        self.phase_embedding_layer = PhaseEmbedding(self.max_sequence_len, self.embedding_dim)
        self.amplitude_embedding_layer = AmplitudeEmbedding(self.embedding_matrix, random_init = False)
        self.complex_embed = ComplexEmbedding(self.embedding_matrix, sentiment_lexicon)
        self.l2_norm = L2Norm(dim = -1, keep_dims = False)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = -1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.proj_measurement = ComplexProjMeasurement(self.embedding_dim)
        self.measurement = ComplexMeasurement(self.embedding_dim, units = 2*self.num_measurements)
        self.use_lexicon_as_measurement = opt.use_lexicon_as_measurement
        self.num_hidden_layers = 3
        
    def forward(self, input_seq):
        """
        In the forward function we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        """
        
        amplitude_embedding, phase_embedding  = self.complex_embed(input_seq)
#        phase_embedding = self.phase_embedding_layer(input_seq)
#        amplitude_embedding = self.amplitude_embedding_layer(input_seq)
        weights = self.l2_norm(amplitude_embedding)
        amplitude_embedding = self.l2_normalization(amplitude_embedding)
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
        post_measurement_states_real = []
        post_measurement_states_imag = []
        prob_list = []
        for n_gram in self.ngram:
            for i in range(self.num_hidden_layers):
                n_gram_embedding_real = n_gram(seq_embedding_real)
                n_gram_embedding_imag = n_gram(seq_embedding_imag)
                n_gram_weight = n_gram(weights)
                n_gram_weight = self.activation(n_gram_weight)
                
                [sentence_embedding_real, sentence_embedding_imag] = self.mixture([n_gram_embedding_real, n_gram_embedding_imag,n_gram_weight])
                output = self.proj_measurement([sentence_embedding_real, sentence_embedding_imag])
                post_measurement_states_real.append(output[0])
                post_measurement_states_imag.append(output[1])
        
                seq_embedding_real = torch.stack(post_measurement_states_real, dim = 1)
                seq_embedding_imag = torch.stack(post_measurement_states_imag, dim = 1)
            
            [sentence_embedding_real, sentence_embedding_imag] = self.mixture([n_gram_embedding_real, n_gram_embedding_imag,n_gram_weight])
            mea_operator = None
            if self.use_lexicon_as_measurement:
                amplitude_measure_operator, phase_measure_operator = self.complex_embed.sample(self.num_measurements)
                mea_operator = self.complex_multiply([phase_measure_operator, amplitude_measure_operator])
            prob_list.append(self.measurement([sentence_embedding_real, sentence_embedding_imag], measure_operator=mea_operator))
            
        probs = torch.stack(prob_list,dim = -1)
        probs_feature = []
        for one_type in self.pooling_type.split(','):
            if self.pooling_type == 'max':
#                probs = GlobalMaxPooling1D()(self.probs)
                # max out the sequence dimension
                probs,_ = torch.max(probs,1,False)
                
            elif self.pooling_type == 'average':
                # average out the sequence dimension
                probs = torch.mean(probs,1,False)
                
            elif self.pooling_type == 'none':
                # do nothing at all, flatten
                probs = torch.flatten(probs, start_dim=1, end_dim=2)
                
            elif self.pooling_type == 'max_col':
                # max out the measurement dimension
                probs,_ = torch.max(probs,2,False)
                
            elif self.pooling_type == 'average_col':
                # average out the measurement dimension
                probs = torch.mean(probs,2,False)
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = torch.flatten(probs, start_dim=1, end_dim=2)
            probs_feature.append(probs)
        
        output = torch.cat(probs_feature,dim = 1)
        output = torch.flatten(probs, start_dim=1, end_dim=2)
#        output = torch.log10(output)
        
        output = nn.Linear(in_features = output.shape[-1], out_features = 16)(output)
        output = nn.Linear(in_features = 16, out_features = 2)(output)
#        output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        
        return output

#    m = nn.MaxPool1d(16)
#    input_tensor = torch.randn(16, 50, 16)
#    output_tensor = m(input_tensor)
#    print(output_tensor.shape)