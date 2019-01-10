# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn

class ComplexMeasurement(torch.nn.Module):
    def __init__(self, embed_dim, units=5, ortho_init=False, device = torch.device('cpu')):
        super(ComplexMeasurement, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        if ortho_init:
            self.kernel = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1))

#            self.real_kernel = torch.nn.Parameter(torch.eye(embed_dim))
#            self.imag_kernel = torch.nn.Parameter(torch.zeros(embed_dim, embed_dim))
        else:
            rand_tensor = torch.rand(self.units, self.embed_dim, 2)
            normalized_tensor = F.normalize(rand_tensor.view(self.units, -1), p=2, dim=1, eps=1e-10).view(self.units, self.embed_dim, 2)
            self.kernel = torch.nn.Parameter(normalized_tensor)
#            self.kernel = F.normalize(self.kernel.view(self.units, -1), p=2, dim=1, eps=1e-10).view(self.units, embed_dim, 2)

#            self.real_kernel = torch.nn.Parameter(torch.Tensor(self.units, embed_dim))
#            self.imag_kernel = torch.nn.Parameter(torch.Tensor(self.units, embed_dim))

    def forward(self, inputs, measure_operator=None):
        
        input_real = inputs[0]
        input_imag = inputs[1]
        
        real_kernel = self.kernel[:,:,0]
        imag_kernel = self.kernel[:,:,1]
        if measure_operator is None:
            real_kernel = real_kernel.unsqueeze(-1)
            imag_kernel = imag_kernel.unsqueeze(-1)
        else:
            real_kernel = measure_operator[0].unsqueeze(-1)
            imag_kernel = measure_operator[1].unsqueeze(-1)

        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
            + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))  
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
            - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # only real part is non-zero
        # input_real.shape = [batch_size, seq_len, embed_dim, embed_dim] or [batch_size, embed_dim, embed_dim]
        # projector_real.shape = [num_measurements, embed_dim, embed_dim]
        output_real = torch.matmul(torch.flatten(input_real, start_dim = -2, end_dim = -1), torch.flatten(projector_real, start_dim = -2, end_dim = -1).t())\
            - torch.matmul(torch.flatten(input_imag, start_dim = -2, end_dim = -1), torch.flatten(projector_imag, start_dim = -2, end_dim = -1).t())
    
        return output_real
    
if __name__ == '__main__':
    model = ComplexMeasurement(6, units=3)
    a = torch.randn(5,6,6)
    b = torch.randn(5,6,6)
#
    y_pred = model([a,b])
    print(y_pred.shape)
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
