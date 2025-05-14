# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth=5e-5):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def perceptual_loss(fake, real):
    return F.l1_loss(fake, real)
