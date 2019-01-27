# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class SGD_Unitary(Optimizer):
    """Implements SGD gradient descent for unitary matrix.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        


    .. note::
        This is the vanilla version of the gradient descent for unitary matrix, 
        i.e. formula (6) in H. D. Tagare. Notes on optimization on Stiefel manifolds. 
        Technical report, Yale University, 2011, and formula (6) in Scott Wisdom, 
        Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity 
        unitary recurrentneural networks. In NIPS 2016. 

        .. math::
                  A = G^H*W - W^H*G \\
                  W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W

        where W, G and lr denote the parameters, gradient
        and learning rate respectively.
    """

    def __init__(self, params, lr_unitary=1e-2, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, device = torch.device('cpu')):
        if not 0.0 <= lr_unitary:
            raise ValueError("Invalid unitary learning rate: {}".format(lr))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr_unitary=lr_unitary, lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.device = device
        super(SGD_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                    d_p = torch.add(torch.zeros_like(p.data), -group['lr'], d_p)
                
                lr_unitary = group['lr_unitary']

                if  torch.cuda.is_available() :
                    G = d_p[:,:,0].cpu().numpy()+1j* d_p[:,:,1].cpu().numpy()               
                    W = p.data[:,:,0].cpu().numpy()+1j* p.data[:,:,1].cpu().numpy()   
                else:
                    G = d_p[:,:,0].numpy()+1j* d_p[:,:,1].numpy()               
                    W = p.data[:,:,0].numpy()+1j* p.data[:,:,1].numpy()  

                #A = G^H W - W^H G
                A_skew = np.matmul(np.matrix.getH(G),W) - np.matmul(np.matrix.getH(W),G)
                
                #(A + i*B)^-1 = (A + B*A^-1*B)^-1 - i*(B + A*B^-1*A)^-1
                #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
                identity = np.eye(A_skew.shape[0])
                cayley_denom =  np.linalg.inv(identity + (lr_unitary/2)* A_skew)
                cayley_numer = identity - (lr_unitary/2)* A_skew
                W_new = np.matmul(np.matmul(cayley_denom,cayley_numer),W)
                
                p_new_real = torch.tensor(W_new.real, dtype = torch.float).to(self.device)
                p_new_imag = torch.tensor(W_new.imag, dtype = torch.float).to(self.device)
                p_new = torch.cat((p_new_real.unsqueeze(2),p_new_imag.unsqueeze(2)),2)

                p.data = p_new


        return loss