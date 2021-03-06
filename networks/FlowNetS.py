'''
Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .submodules import *
'Parameter count : 38,676,504 '

class FlowNetS(nn.Module):
    def __init__(self, args, input_channels=12, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1    = conv(self.batchNorm,  input_channels, 16, kernel_size=7, stride=2)

        self.conv2    = conv(self.batchNorm, 16, 32, kernel_size=5, stride=2)
        self.conv3    = conv(self.batchNorm, 32, 64, kernel_size=5, stride=2)
        self.conv3_1  = conv(self.batchNorm, 64, 64)

        self.conv4    = conv(self.batchNorm, 64, 128, stride=2)
        self.conv4_1  = conv(self.batchNorm, 128, 128)
#         self.conv5    = conv(self.batchNorm, 128, 128, stride=2)
#         self.conv5_1  = conv(self.batchNorm, 128, 128)
#         self.conv6    = conv(self.batchNorm, 128, 256, stride=2)
#         self.conv6_1  = conv(self.batchNorm, 256, 256)

#         self.deconv5 = deconv(256, 128)
#         self.deconv4 = deconv(258, 64)
        self.deconv3 = deconv(128, 32)
        self.deconv2 = deconv(32+64+2, 16)

#         self.predict_flow6 = predict_flow(256)
#         self.predict_flow5 = predict_flow(258)
        self.predict_flow4 = predict_flow(128)
        self.predict_flow3 = predict_flow(32+64+2)
        self.predict_flow2 = predict_flow(16+32+2)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.dropout = nn.Dropout(0.8)
        
    def forward(self, x):
        out_conv1  = self.conv1(x)
        out_conv1d = self.dropout(out_conv1)
        
        out_conv2  = self.conv2(out_conv1d)
        out_conv2d = self.dropout(out_conv2d)
        
        out_conv3  = self.conv3_1(self.conv3(out_conv2d))
        out_conv3d = self.dropout(out_conv3d)
        
        out_conv4  = self.conv4_1(self.conv4(out_conv3d))
        out_conv4.weight = out_conv3.weight.t()
#         out_conv4d = self.dropout(out_conv4)
#         out_conv5  = self.conv5_1(self.conv5(out_conv4d))
#         out_conv5d = self.dropout(out_conv5)
#         out_conv6  = self.conv6_1(self.conv6(out_conv5d))

#         flow6 = self.predict_flow6(out_conv6)
#         flow6_up = self.upsampled_flow6_to_5(flow6)
#         out_deconv5 = self.deconv5(out_conv6)

#         concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
#         flow5 = self.predict_flow5(concat5)
#         flow5_up = self.upsampled_flow5_to_4(flow5)

#         out_deconv4 = self.deconv4(concat5)

#         concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
#         flow4 = self.predict_flow4(concat4)
        flow4 = self.predict_flow(out_conv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(out_conv4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4 #, flow5, flow6
        else:
            return flow2,
