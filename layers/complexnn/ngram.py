# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np

class NGram(torch.nn.Module):
    '''
    Input can be a sequence of indexes or a sequence of embeddings
    gram_n is the value of n
    dim is the dimension to which n-gram is applied
    out_n = seq_len+(n_gram-1)*2-n_gram +1 = 
    e.g. input_shape = (None,10) gram_n = 5 ==> output_shape = (None,10,5)
    e.g. input_shape = (None,10,3) gram_n = 5, axis = 1 ==> output_shape = (None,10,5,3)
    
    '''
    def __init__(self, gram_n=3, dim=1, device = torch.device('cpu')):
        super(NGram, self).__init__()
        self.gram_n = gram_n
        self.dim = dim
        self.device = device
        
    def forward(self, inputs):
        
        slice_begin_index = 0
        slice_end_index = -1

        batch_size, seq_len, embed_dim = inputs.shape
        total_padded_len = self.gram_n - 1
        left_padded_len = int(total_padded_len/2)
        right_padded_len = total_padded_len - left_padded_len
        left_padded_zeros = torch.zeros(batch_size, left_padded_len, embed_dim).to(self.device)
        right_padded_zeros = torch.zeros(batch_size, right_padded_len, embed_dim).to(self.device)
        
        inputs = torch.cat([left_padded_zeros, inputs, right_padded_zeros], dim=self.dim)

        out_n = seq_len 
        list_of_ngrams = []
        
        for i in range(out_n):
            slice_begin_index = i
            slice_end_index = i + self.gram_n
            slice_index = torch.tensor(np.arange(slice_begin_index, slice_end_index), dtype=torch.long).to(self.device)
            l = torch.index_select(inputs, self.dim, index=slice_index)
            list_of_ngrams.append(torch.unsqueeze(l, dim=self.dim+1))
                
        ngram_mat = torch.cat(list_of_ngrams, dim=self.dim+1)
        
        return ngram_mat

def test():
    n_gram = NGram(gram_n = 4)
    a = torch.LongTensor(2,5).random_(0, 10)
    n_gram_mat = n_gram(a)
    print(n_gram_mat)
    if n_gram_mat.dim() == a.dim() + 1:
        print('NGram Test Passed.')
    else:
        print('NGram Test Failed.')

if __name__ == '__main__':
    test()