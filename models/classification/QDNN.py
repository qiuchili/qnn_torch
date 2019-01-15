# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.complexnn import *

class QDNN(torch.nn.Module):
    def __init__(self, opt):
        """
        max_sequence_len: input sentence length
        embedding_dim: input dimension
        num_measurements: number of measurement units, also the output dimension

        """
        super(QDNN, self).__init__()
        self.max_sequence_len = opt.max_sequence_length
        self.device = opt.device
        sentiment_lexicon = opt.sentiment_dic
        if sentiment_lexicon is not None:
            sentiment_lexicon = torch.tensor(sentiment_lexicon, dtype=torch.float)
        self.num_measurements = opt.measurement_size
        self.embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embedding_dim = self.embedding_matrix.shape[1]
        self.complex_embed = ComplexEmbedding(opt,self.embedding_matrix, sentiment_lexicon)
        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = 1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.measurement = ComplexMeasurement(self.embedding_dim, units = 2*self.num_measurements,device = self.device)
        self.dense = nn.Linear(in_features = 2*self.num_measurements, out_features = 2)
        self.use_lexicon_as_measurement = opt.use_lexicon_as_measurement

        
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
        weights = self.activation(weights)
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
        [sentence_embedding_real, sentence_embedding_imag] = self.mixture([seq_embedding_real, seq_embedding_imag,weights])
        
        mea_operator = None
        if self.use_lexicon_as_measurement:
            amplitude_measure_operator, phase_measure_operator = self.complex_embed.sample(self.num_measurements)
            mea_operator = self.complex_multiply([phase_measure_operator, amplitude_measure_operator])
        
        output = self.measurement([sentence_embedding_real, sentence_embedding_imag], measure_operator=mea_operator)
#        output = torch.log10(output)
        output = self.dense(output)
#        output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        
        return output