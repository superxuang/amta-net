#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(torch.mul(predict, target))*2 + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth
        dice = intersection / union

        loss = 1 - dice
        return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=[], **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        if weight is not None:
            self.weight = weight / weight.sum()
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        total_loss_num = 0

        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
                total_loss_num += 1

        if self.weight is not None:
            return total_loss
        elif total_loss_num > 0:
            return total_loss/total_loss_num
        else:
            return 0