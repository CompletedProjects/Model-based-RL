#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2dSame(torch.nn.Module): # Nawid - Performs convolution in the same way as 'same' tensorflow format I assume
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out

class Encoder(nn.Module): # Nawid -  CNN architecture in the Impala paper
    def __init__(self, obs_shape, feature_dim, downsample = True):
        super(Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.depths = [16, 32, 32, 32]
        self.downsample = downsample
        '''
        self.encoder_sizes = [obs_shape[2],self.depths[0],self.depths[1],self.depths[2],self.depths[3]]
        conv_blocks = [self._make_layer(in_channels, out_channels)
                       for in_channels, out_channels in zip(self.encoder_sizes, self.encoder_sizes[1:])]
        self.conv_encoder = nn.Sequential(*conv_blocks)
        '''
        self.layer1 = self._make_layer(obs_shape[2], self.depths[0]) # obs shape is the original numpy ordering so I placing the channel dimension into the network
        self.layer2 = self._make_layer(self.depths[0], self.depths[1])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2])
        self.layer4 = self._make_layer(self.depths[2], self.depths[3])

        if self.downsample:
            self.final_conv_size = 32 * 4 * 4
        else:
            self.final_conv_size = 32 * 9 * 9

        self.final_linear = nn.Linear(self.final_conv_size, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim) # Nawid -  Layer norm used instead of batch norm due to statistic sharing
        self.flatten = Flatten()
        self.train()

    def _make_layer(self, in_channels, depth): # Nawid-  Used to make a layer
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),# Changed the stride to 1 - originally it was 1, larger stride more information can be lost but larger effective window size to model long range interactions
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.

        #f5 = self.conv_encoder[0:-1](obs)
        f5 = self.layer3(self.layer2(self.layer1(obs))) # Nawid -Uses the output of the third layer and then sees whether we want to downsample or
        if self.downsample:
            out = self.layer4(f5) #self.conv_encoder[-1](f5)
        else:
            out = f5

        h = self.flatten(out)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs) # Nawid - Performs conv layers
        if detach:
            h = h.detach()

        h_fc = self.final_linear(h)
        h_norm = self.ln(h_fc) # Nawid - Passes through layer normalisation

        return h_norm

def make_encoder(obs_shape, feature_dim, downsample = True):
    return Encoder(
        obs_shape, feature_dim, downsample
        )
