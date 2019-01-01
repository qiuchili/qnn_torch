# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn
from torch.optim import SGD
from optimizer.pytorch_optimizer import Vanilla_Unitary
from torch.distributions import Categorical
class ComplexProjMeasurement(torch.nn.Module):
    def __init__(self, embed_dim):
        super(ComplexProjMeasurement, self).__init__()
#        self.units = units
        self.embed_dim = embed_dim
        self.kernel_real = torch.eye(embed_dim).unsqueeze(2)
        self.kernel_imag = torch.zeros(embed_dim,embed_dim).unsqueeze(2)
        self.kernel = torch.nn.Parameter(torch.cat((self.kernel_real,self.kernel_imag),2))
#        self.kernel = torch.nn.Parameter()
        
#        self.kernel = F.normalize(kernel.view(self.units, -1), p=2, dim=1, eps=1e-10).view(self.units, embed_dim, 2)

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        
        kernel_real = self.kernel[:,:,0] #(embed_dim,embed_dim)
        kernel_imag = self.kernel[:,:,1] #(embed_dim,embed_dim)

#        batch_size = inputs[0].size(0)
    
        
        input_real = inputs[0] #(batch_size,embed_dim,embed_dim)
        input_imag = inputs[1] #(batch_size,embed_dim,embed_dim)
        batch_size = input_real.shape[0]
#        print(torch.matmul(kernel_real,kernel_real)+torch.matmul(kernel_imag,kernel_imag))
        
        
        if input_real.dim() == 3:
            dim = input_real.shape[-1]      
            output = torch.zeros(batch_size, dim,2)
            for j in range(batch_size):
                prob_distribution = torch.zeros(dim)
                for i in range(dim):
                    v_real = kernel_real[i,:]
                    v_imag = kernel_imag[i,:]
                    proj_real = torch.matmul(v_real.view(dim,1),v_real.view(1,dim))+torch.matmul(v_imag.view(dim,1),v_imag.view(1,dim))
                    proj_imag = torch.matmul(v_real.view(dim,1),v_imag.view(1,dim))-torch.matmul(v_imag.view(dim,1),v_real.view(1,dim))
                    multiplication = torch.matmul(input_real,proj_real.view(1,dim,dim))-torch.matmul(input_imag,proj_imag.view(1,dim,dim))
                    prob_distribution[i]=torch.trace(multiplication[j,:,:])
                    
                m = Categorical(prob_distribution)           
                print(prob_distribution)
                index = m.sample()
                output[j,:,0] = kernel_real[index,:]
                output[j,:,1] = kernel_imag[index,:]
            output = [output[:,:,0],output[:,:,1]]
            
        elif input_real.dim() == 4:
            dim = input_real.shape[-1]
            seq_len = input_real.shape[1]
            output = torch.zeros(batch_size,seq_len,dim,2)
            for j in range(batch_size):
                for k in range(seq_len):
                    prob_distribution = torch.zeros(dim)
                    for i in range(dim):
                        v_real = kernel_real[i,:]
                        v_imag = kernel_imag[i,:]
                        proj_real = torch.matmul(v_real.view(dim,1),v_real.view(1,dim))+torch.matmul(v_imag.view(dim,1),v_imag.view(1,dim))
                        proj_imag = torch.matmul(v_real.view(dim,1),v_imag.view(1,dim))-torch.matmul(v_imag.view(dim,1),v_real.view(1,dim))
                        multiplication = torch.matmul(input_real,proj_real.view(1,dim,dim))-torch.matmul(input_imag,proj_imag.view(1,dim,dim))
                        prob_distribution[i]=torch.trace(multiplication[j,k,:,:])
                    
                    m = Categorical(prob_distribution)           
#                    print(prob_distribution)
                    index = m.sample()
                    output[j,k,:,0] = kernel_real[index,:]
                    output[j,k,:,1] = kernel_imag[index,:]
            output = [output[:,:,0],output[:,:,1]]

                    
        
    
#        output_imag = torch.matmul(input_real,kernel_imag) + torch.matmul(input_imag, kernel_real)
        
        
#        projector_real = torch.bmm(kernel_real, kernel_real.transpose(1, 2)) \
#            + torch.bmm(kernel_imag, kernel_imag.transpose(1, 2))
#            
#        projector_imag = torch.bmm(kernel_imag, kernel_real.transpose(1, 2)) \
#            - torch.bmm(kernel_real, kernel_imag.transpose(1, 2))
#
#        output_real = torch.mm(input_real.view(batch_size, self.embed_dim*self.embed_dim), projector_real.view(self.units,self.embed_dim*self.embed_dim).t())\
#        - torch.mm(input_imag.view( batch_size, self.embed_dim*self.embed_dim), projector_imag.view(self.units,self.embed_dim*self.embed_dim).t())
        
#        output_imag = torch.mm(input_real.view(batch_size, self.embed_dim*self.embed_dim), projector_imag.view(self.units,self.embed_dim*self.embed_dim).t())\
#        + torch.mm(input_imag.view(batch_size, self.embed_dim*self.embed_dim), projector_real.view(self.units,self.embed_dim*self.embed_dim).t())
          
        return output
    
if __name__ == '__main__':
  
    model = ComplexProjMeasurement(100)
    loss_function = torch.nn.MSELoss()
    optimizer = Vanilla_Unitary(model.parameters(), lr = 0.01)
    
    a = torch.randn(5,10,100,100)
    b = torch.randn(5,10,100,100)
    c = torch.randn(5,10,100)
    losses = []
    for epoch in range(100):
       
        x_input = [a,b]
        optimizer.zero_grad()
        y_pred = model(x_input)
        loss = loss_function(y_pred, c)  
        loss.backward()
        optimizer.step()
        total_loss = loss.item()
        losses.append(total_loss)
        
    print(losses)
    
    
    