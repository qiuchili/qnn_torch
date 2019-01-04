# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn

class ComplexMeasurement(torch.nn.Module):
    def __init__(self, embed_dim, units=5, ortho_init=False):
        super(ComplexMeasurement, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        if ortho_init:
            self.real_kernel = torch.nn.Parameter(torch.eye(embed_dim))
            self.imag_kernel = torch.nn.Parameter(torch.zeros(embed_dim, embed_dim))
        else:
            self.real_kernel = torch.nn.Parameter(torch.Tensor(self.units, embed_dim))
            self.imag_kernel = torch.nn.Parameter(torch.Tensor(self.units, embed_dim))

    def forward(self, inputs, measure_operator=None):
        batch_size = inputs[0].size(0)
        input_real = inputs[0]
        input_imag = inputs[1]
        
        if measure_operator is None:
            real_kernel = self.real_kernel.unsqueeze(-1)
            imag_kernel = self.imag_kernel.unsqueeze(-1)
        else:
            real_kernel = measure_operator[0].unsqueeze(-1)
            imag_kernel = measure_operator[1].unsqueeze(-1)

        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
            + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))  
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
            - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # only real part is non-zero
        output_real = torch.mm(input_real.view(batch_size, self.embed_dim*self.embed_dim), projector_real.view(self.units,self.embed_dim*self.embed_dim).t())\
            - torch.mm(input_imag.view(batch_size, self.embed_dim*self.embed_dim), projector_imag.view(self.units,self.embed_dim*self.embed_dim).t())
    
        return output_real
    
if __name__ == '__main__':
    model = ComplexMeasurement(6, units=3)
    a = torch.randn(5,6,6)
    b = torch.randn(5,6,6)

    y_pred = model([a,b])
    print(y_pred.shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    