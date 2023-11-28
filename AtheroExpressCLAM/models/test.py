import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils.utils import initialize_weights
import numpy as np


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        print(f'a shape : {a.shape} ')
        b = self.attention_b(x)
        print(f'b shape : {b.shape} ')
        A = a.mul(b)
        print(f'A shape : {A.shape} ')
        A = self.attention_c(A)  # N x n_classes
        print(f'A shape : {A.shape} ')
        return A, x


attention_net = Attn_Net_Gated(L = 512, D = 256, dropout = 0.3, n_classes = 1)
import torch

v = torch.ones((1, 100, 512))
attention_net(v)
