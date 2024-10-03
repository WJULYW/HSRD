# -*- coding: UTF-8 -*-
from basic_module import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import utils
from torchvision import models
import numpy as np

np.set_printoptions(threshold=np.inf)
sys.path.append('..')


class Discriminator(nn.Module):
    def __init__(self, max_iter, domain_num=4):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, domain_num)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out


class HSRD(nn.Module):
    def __init__(self, ada_num=2, max_iter=4000, domain_num=4):
        super(HSRD, self).__init__()
        self.encoder = BaseNet()

        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(256) for i in range(ada_num)])

        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        self.FC = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256)
        )
        self.dis = Discriminator(max_iter)

        self.fc = nn.Linear(512, 1)


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 128, [1, 1], downsample=1),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(128, 64, [2, 1], downsample=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [1, 1], downsample=1),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 16, [2, 1], downsample=1),
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(16, 1, [1, 1], downsample=1),
        )

    def cal_gamma_beta(self, x1):

        embs = self.encoder.get_rep(x1)
        x1_4 = embs[0]

        x1_add = embs[1]
        x1_add = self.ada_conv1(x1_add) + embs[2]
        x1_add = self.ada_conv2(x1_add) + embs[3]
        x1_add = self.ada_conv3(x1_add)

        gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)
        gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)

        domain_invariant = torch.nn.functional.adaptive_avg_pool2d(x1_4, 1).reshape(x1_4.shape[0], -1)

        return x1_4, gamma, beta, domain_invariant

    def forward(self, input, input2):
        x1, gamma1, beta1, domain_invariant = self.cal_gamma_beta(input)
        x2, gamma2, beta2, _ = self.cal_gamma_beta(input2)

        fea_x1_x1 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)

        fea_x1_x1 = self.conv_final(fea_x1_x1)
        em = self.avgpool(fea_x1_x1).view(fea_x1_x1.shape[0], -1)

        HR = self.fc(em)
        # For Sig
        x = self.up1(fea_x1_x1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        Sig = self.up6(x).squeeze(dim=1)
        fea_x1_x1 = em


        fea_x1_x2 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x2 = self.adaIN_layers[i](fea_x1_x2, gamma2, beta2)
        fea_x1_x2 = self.conv_final(fea_x1_x2)
        fea_x1_x2 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x2, 1)
        fea_x1_x2 = fea_x1_x2.reshape(fea_x1_x2.shape[0], -1)

        dis_invariant = self.dis(domain_invariant)
        return HR, Sig, fea_x1_x1, fea_x1_x2, dis_invariant

