import torch
import torch.nn as nn
import torch.utils.data as data_utl
import os
from PIL import Image
import numpy as np
import random
import pandas as pd

#penalty loss만들기################################################################
class PenaltyLoss(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func
    def forward(self, outputs, label, preds):
        label_ind = label.nonzero()
        pred_ind = preds.nonzero()
        incorrect = 0
        if label.sum():
            SL = min(label_ind)
            EL = max(label_ind)
            if pred_ind.sum():
                for i in pred_ind:
                    if i < SL:
                        incorrect = incorrect + SL-i
                    elif i > EL:
                        incorrect = incorrect + i - EL
            else: 
                incorrect = max(SL, len(label)-EL)*len(label_ind)
        return self.loss_func(outputs, label)+incorrect


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Both1Lin(nn.Module):
    def __init__(self, forward_model, backward_model, cat_length, num_classes=2):
        super().__init__()
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.cat = nn.Linear(cat_length, num_classes)
    def forward(self, forward, backward, cat_dim=1):
        forward = self.forward_model(forward)
        backward = self.backward_model(backward)
        #[batch, features] 일 경우, cat_dim=1
        # print(forward.size())
        x = torch.cat((forward, backward), dim=cat_dim)
        if (len(x.size())==3) and (x.size(-1)==1): # if i3d model
            x = x.squeeze(2)
        # print(x.size())
        x = self.cat(x)
        # print(x.size())
        return x

class MHA_3D(nn.Module): # Multi Head Attention 3D
    def __init__(self, embed_dim, num_heads, num_classes):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, src, mask=None):
        src, _ = self.attn(src, src, src, key_padding_mask=mask) 
        src = self.fc(src)
        return src
    
class PE(nn.Module):
    "Implement the positional encoding function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        import math
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [[1], [2], ..., [max_len]]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, start_f=None):
        if start_f is not None:
            f = x.size(1)
            for b in range(x.size(0)):
                x[b] = x[b]+self.pe[:, start_f[b] : start_f[b] + f].requires_grad_(False)
        else:
            x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)      

class PeB4Fc(nn.Module): 
    def __init__(self, backbone, PE, FC, freeze=False):
        super(PeB4Fc, self).__init__()
        self.backbone = backbone
        self.PE = PE
        self.FC = FC
        self.freeze = freeze # whether freeze backbone module
    
    def forward(self, x, start_f=None, mask=None):
        if self.freeze:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        x = self.PE(x, start_f)
        x = self.FC(x, mask)
        return x

# https://github.com/abhuse/polyloss-pytorch/blob/main/polyloss.py
import torch.nn.functional as F
from torch import Tensor
class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: Tensor = None,
                 pos_weight: Tensor = None,
                 label_is_onehot: bool = True):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1
