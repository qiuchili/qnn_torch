# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class RMSprop_Unitary(Optimizer):
    """Implements RMSprop gradient descent for unitary matrix.
        
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

    def __init__(self, params, lr_unitary=1e-2,  lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, device = torch.device('cpu')):
        if not 0.0 <= lr_unitary:
            raise ValueError("Invalid unitary learning rate: {}".format(lr))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        self.device = device
        defaults = dict(lr_unitary=lr_unitary, momentum = momentum, lr=lr, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

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
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                
                if group['momentum'] >0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    d_p = torch.add(torch.zeros_like(p.data), -group['lr'], buf)
#                    p.data.add_(-group['lr'], buf)
                else:
                    d_p = torch.addcdiv(torch.zeros_like(p.data), -group['lr'], grad, avg)
#                    p.data.addcdiv_(-group['lr'], grad, avg)
#             
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


