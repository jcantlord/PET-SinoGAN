"""
Critic and Generator implementation from WGAN-GP paper

Based on https://github.com/aladdinpersson
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            # Input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),  # 64x64
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 32x32
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=1, padding=0),  # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1 
            self._block(z_dim, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g*8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g*4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g*2, 4, 2, 1),  # img: 32x32
            self._block(features_g * 2, features_g, 4, 2, 1),  # img: 64x64x64
            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1,
            ),
            # Output: N x channels_img x 128 x 128
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    # Initializes weights according to DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
