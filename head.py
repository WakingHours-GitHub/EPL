from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import numpy as np


class Cosface(nn.Module):
    def __init__(self, in_features, out_features, m=0.0, s=64):
        super(Cosface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, input, label, epoch=None):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        d_theta = one_hot * self.m
        logits = self.s * (cos_theta - d_theta) # cosface.
        losses = F.cross_entropy(logits, label, reduction='none')
        return losses.mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'

        
        

class Arcface(nn.Module):
    def __init__(self, in_features, out_features, m=0.5, s=64):
        super(Arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.mode = "easy_margin"
        
        
    def forward(self, input, label, epoch=None):
        cosine = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))
        cosine.clamp_(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        threshold = math.cos(math.pi - self.m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        cosine_m = cosine * math.cos(self.m) - sine * math.sin(self.m)
        if self.mode == "easy_margin":
            phi = torch.where(cosine > 0, cosine_m, cosine)
        elif self.mode == "hard_margin":
            phi = torch.where(cosine > threshold, cosine_m, cosine - self.m)
        elif self.mode == "characteristic_gradient_detachment":
            phi = cosine - cosine.detach() + cosine_m.detach()
        else:
            phi = cosine_m
            
        logits = torch.where(one_hot, phi, cosine)
        logits *= self.s
        arcface_loss = F.cross_entropy(logits, label)

        return arcface_loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + 'mode=' + str(self.mode) + ')'

class Unified_Cross_Entropy_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64):
        super(Unified_Cross_Entropy_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*10))

    def forward(self, input, label, epoch=None):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5)) # [B, C]
        

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s)))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        losses = one_hot * p_loss + (~one_hot) * n_loss
        return losses.sum(dim=1).mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'
