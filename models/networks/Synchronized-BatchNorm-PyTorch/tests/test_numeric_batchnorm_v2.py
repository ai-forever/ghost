#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_numeric_batchnorm_v2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/01/2018
#
# Distributed under terms of the MIT license.

"""
Test the numerical implementation of batch normalization.

Author: acgtyrant.
See also: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
"""

import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from sync_batchnorm.unittest import TorchTestCase
from sync_batchnorm.batchnorm_reimpl import BatchNorm2dReimpl


class NumericTestCasev2(TorchTestCase):
    def testNumericBatchNorm(self):
        CHANNELS = 16
        batchnorm1 = nn.BatchNorm2d(CHANNELS, momentum=1)
        optimizer1 = optim.SGD(batchnorm1.parameters(), lr=0.01)

        batchnorm2 = BatchNorm2dReimpl(CHANNELS, momentum=1)
        batchnorm2.weight.data.copy_(batchnorm1.weight.data)
        batchnorm2.bias.data.copy_(batchnorm1.bias.data)
        optimizer2 = optim.SGD(batchnorm2.parameters(), lr=0.01)

        for _ in range(100):
            input_ = torch.rand(16, CHANNELS, 16, 16)

            input1 = input_.clone().requires_grad_(True)
            output1 = batchnorm1(input1)
            output1.sum().backward()
            optimizer1.step()

            input2 = input_.clone().requires_grad_(True)
            output2 = batchnorm2(input2)
            output2.sum().backward()
            optimizer2.step()

        self.assertTensorClose(input1, input2)
        self.assertTensorClose(output1, output2)
        self.assertTensorClose(input1.grad, input2.grad)
        self.assertTensorClose(batchnorm1.weight.grad, batchnorm2.weight.grad)
        self.assertTensorClose(batchnorm1.bias.grad, batchnorm2.bias.grad)
        self.assertTensorClose(batchnorm1.running_mean, batchnorm2.running_mean)
        self.assertTensorClose(batchnorm2.running_mean, batchnorm2.running_mean)


if __name__ == '__main__':
    unittest.main()

