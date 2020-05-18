#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
def tie_weights(src, trg): # Nawid - Used to make the weights between 2 different networks be the same, so if one is updated the other is updated I believe, this is different than just copying the weights once
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList( # Nawid - 3 conv layers in total with output dim 32
            [nn.Conv2d(obs_shape[2], num_filters, 3, stride=2)] # Nawid - Changed this to take into account original numpy ordering
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-2] == 64 else OUT_DIM[num_layers] # changed obs_shape[1] to -2 to take into account numpy ordering
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim) # Nawid -  Layer norm used instead of batch norm due to statistic sharing

        self.output_logits = output_logits

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.
        #obs = obs / 255. #Changed it so that the observations are divided
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        #h = conv.view(conv.size(0), -1)
        h = conv.reshape([conv.size(0),-1])
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs) # Nawid - Performs conv layers
        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc) # Nawid - Passes through layer normalisation

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])



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


class ImpalaEncoder(nn.Module): # Nawid -  CNN architecture in the Impala paper
    def __init__(self, obs_shape, feature_dim, downsample = True):
        super(ImpalaEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.depths = [16, 32, 32, 32]
        self.downsample = downsample
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

    def forward(self, inputs):
        if inputs.max() > 1.:
            inputs = inputs / 255.
        #f1 = self.layer1(inputs)
        #f2 = self.layer2(f1)
        #f5 = self.layer3(f2)
        f5 = self.layer3(self.layer2(self.layer1(inputs))) # Nawid -Uses the output of the third layer and then sees whether we want to downsample or
        if self.downsample:
            out = self.layer4(f5)
        else:
            out = f5

        out = self.final_linear(self.flatten(out)) # Nawid- global feature vector - replaced the relu with a layernorm similar to what was done in curl, i do not generally see an activation function after the final layer so I removed the relu
        out = self.ln(out)
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'Impala': ImpalaEncoder}

def make_encoder(encoder_type, obs_shape, feature_dim, num_layers=4, num_filters=32, output_logits=False, downsample = True):
    assert encoder_type in _AVAILABLE_ENCODERS
    if encoder_type =='pixel':
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters, output_logits
            )
    else:
        return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, downsample
        )
