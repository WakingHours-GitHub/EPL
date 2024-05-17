from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import numpy as np

class EPL(nn.Module):
    def __init__(self, reduction='mean'):
        super(EPL, self).__init__()
        self.reduction = reduction

    def forward(self, output, labels):
        labels = labels.long()
        sum_neg = ((1 - labels) * torch.exp(output)).sum(1)
        sum_pos = (labels * torch.exp(-output)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

class CosFace_EPL(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, k=0.7):
        super(CosFace_EPL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

        # EPL:
        self.k = k
        self.classes = out_features
        self.start_vp_epoch = 4
        self.update_criteria = F.softsign
        self.criteria = EPL()
        print(self.criteria)
        print("self.start_vp_epoch:", self.start_vp_epoch, "k:", self.k)
        print(self.update_criteria)
        self.register_buffer("queue", torch.randn(out_features, in_features, requires_grad=False))

    def update_virtual_prototype(self, embeddings, labels):
        embeddings = embeddings.clone().detach() # detach() -> no have grident.
        embeddings = F.normalize(embeddings, eps=1e-5)
        feature_drift_qeueu_embedding = torch.diag(self.queue[labels] @ embeddings.T)
        # factor = F.sigmoid(feature_drift_qeueu_embedding-1).unsqueeze(1)
        if "sigmoid" in self.update_criteria.__str__():
            factor = F.sigmoid(feature_drift_qeueu_embedding-1).unsqueeze(1)
        elif "direct"== self.update_criteria:
            factor = 0
        else:
            factor = self.update_criteria(feature_drift_qeueu_embedding).unsqueeze(1)
        self.queue[labels] = F.normalize(factor * self.queue[labels] + (1 - factor) * embeddings)


    def forward(self, input, label, epoch=None):
        embeddings = F.normalize(input, eps=1e-5)
        labels = label.long()
        
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        d_theta = one_hot * self.m
        cosine = cos_theta - d_theta # 还是角度相似度


        # EPL
        self.update_virtual_prototype(embeddings, labels)
        if epoch + 1>= self.start_vp_epoch:
            vp_sim = torch.einsum("be,ce->bc ", [embeddings, F.normalize(self.queue, dim=1).clone().detach()])
            B,C = vp_sim.shape
            vp_sim[torch.arange(B), labels] -= self.k * vp_sim[torch.arange(B), labels].data
            cosine = torch.cat([cosine, vp_sim], dim=0)
            labels = one_hot.long()
            labels = torch.cat([labels, labels], dim=0)
            
            logits = self.s * cosine
            losses = self.criteria(logits, labels)
            return losses
        
        logits = self.s * cosine
        
        losses = self.criteria(logits, one_hot)
        
        return losses

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'


class Arcface_EPL(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, k=0.7):
        super(Arcface_EPL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.mode = "characteristic_gradient_detachment"
        print(self.mode)
        self.k = k
        self.classes = out_features
        self.start_vp_epoch = 4
        self.update_criteria = F.softsign
        
        self.criteria = EPL()
        print(self.criteria)
        print(self.update_criteria)
        print("self.start_vp_epoch:", self.start_vp_epoch, "k:", self.k)
        
        self.register_buffer("queue", torch.randn(out_features, in_features, requires_grad=False))
        
        
    def forward(self, input, labels, epoch=None):
        embeddings = F.normalize(input, eps=1e-5)
        
        cosine = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))
        cosine.clamp_(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        
        self.update_virtual_prototype(embeddings, labels)
        if epoch +1 >= self.start_vp_epoch:
            vp_sim = torch.einsum("be,ce->bc ", [embeddings, F.normalize(self.queue, dim=1).clone().detach()])
            B,C = vp_sim.shape
            vp_sim[torch.arange(B), labels] -= self.k * vp_sim[torch.arange(B), labels].data
            cosine = torch.cat([cosine, vp_sim], dim=0)
            labels = torch.cat([labels, labels], dim=0)
    
        one_hot = torch.zeros((labels.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            
        threshold = math.cos(math.pi - self.m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        cosine_m = cosine * math.cos(self.m) - sine * math.sin(self.m)
        if self.mode == "easy_margin":
            phi = torch.where(cosine > 0, cosine_m, cosine)
        elif self.mode == "hard_margin":
            phi = torch.where(cosine > threshold, cosine_m, cosine - self.mm)
        elif self.mode == "characteristic_gradient_detachment":
            phi = cosine - cosine.detach() + cosine_m.detach()
        else:
            phi = cosine_m
            
        logits = torch.where(one_hot, phi, cosine.float())

        logits *= self.s
        arcface_loss = self.criteria(logits, one_hot.long())

        return arcface_loss
        
    def update_virtual_prototype(self, embeddings, labels):
        embeddings = embeddings.clone().detach() # detach()就不加入到计算图当中去了. 也就是阻碍了反向传播. 
        embeddings = F.normalize(embeddings, eps=1e-5)
        
        feature_drift_qeueu_embedding = torch.diag(self.queue[labels] @ embeddings.T)
        if "sigmoid" in self.update_criteria.__str__():
            factor = F.sigmoid(feature_drift_qeueu_embedding-1).unsqueeze(1)
            print("sigmoid")
        else:
            factor = self.update_criteria(feature_drift_qeueu_embedding).unsqueeze(1)
        
        self.queue[labels] = F.normalize(factor * self.queue[labels] + (1 - factor) * embeddings)
                
        
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + 'mode=' + str(self.mode) + ')'
               

class Uniface_EPL(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, k=0.7):
        super(Uniface_EPL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.l = l
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*10))
        
        self.k = k
        self.classes = out_features
        self.start_vp_epoch = 4
        self.update_criteria = F.softsign
        self.criteria = EPL()
        print(self.criteria)
        print(self.update_criteria)
        print("self.start_vp_epoch", self.start_vp_epoch, "k: ", self.k)
        self.register_buffer("queue", torch.randn(out_features, in_features, requires_grad=False))
        
    def forward(self, input, label, epoch=None):
        embeddings = F.normalize(input, eps=1e-5)
        labels = label.long()
        cosine = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))
        
        self.update_virtual_prototype(embeddings, labels)
        if epoch+1 >= self.start_vp_epoch:
        # if epoch >= 0:
            vp_sim = torch.einsum("be,ce->bc ", [embeddings, F.normalize(self.queue, dim=1).clone().detach()])
            B,C = vp_sim.shape
            vp_sim[torch.arange(B), labels] -= self.k * vp_sim[torch.arange(B), labels].data

            cosine = torch.cat([cosine, vp_sim], dim=0)
            labels = torch.cat([labels, labels], dim=0)

        cos_m_theta_p = self.s * (cosine - self.m) - self.bias
        cos_m_theta_n = self.s * cosine - self.bias
        
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l # 平衡。

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((labels.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        losses = one_hot * p_loss + (~one_hot) * n_loss
        return losses.sum(dim=1).mean()
    
    def update_virtual_prototype(self, embeddings, labels):
        embeddings = embeddings.clone().detach() # detach()就不加入到计算图当中去了. 也就是阻碍了反向传播. 
        embeddings = F.normalize(embeddings, eps=1e-5)
        
        feature_drift_qeueu_embedding = torch.diag(self.queue[labels] @ embeddings.T)
        if "sigmoid" in self.update_criteria.__str__():
            factor = F.sigmoid(feature_drift_qeueu_embedding-1).unsqueeze(1)
        else:
            factor = self.update_criteria(feature_drift_qeueu_embedding).unsqueeze(1)
        
        self.queue[labels] = F.normalize(factor * self.queue[labels] + (1 - factor) * embeddings)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'
