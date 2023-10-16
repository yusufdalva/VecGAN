import os
import sys

from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models

import math

"""
Model definition for the VecGAN generator.
:param hyperparameters: model parameters retrieved from the config file, parsed by parse_config in core/utils.py
:param random_init: True if latent directions are initialized randomly, 
                if False still randomly generated but initial weights close to zero. 
                Default is True.
"""
class Generator(nn.Module):
    def __init__(self, hyperparameters, random_init=True):
        super(Generator, self).__init__()

        # Registering hyperparameters
        self.tags = hyperparameters["tags"]
        self.encoder_channels = hyperparameters["encoder_channels"]
        self.decoder_channels = hyperparameters["decoder_channels"]
        self.latent_dim = hyperparameters["latent_dim"]

        # Defining model parts
        self.encoder = nn.Sequential(
            nn.Conv2d(hyperparameters["channel_size"], self.encoder_channels[0], 1, 1, 0),
            *[DownBlockIN(self.encoder_channels[i], self.encoder_channels[i+1]) for i in range(len(self.encoder_channels) - 2)],
            DownBlock(self.encoder_channels[len(self.encoder_channels) - 2], self.encoder_channels[len(self.encoder_channels) - 1])
        ) # Encoder to extract latent code from input image

        self.decoder = nn.Sequential(
            UpBlock(self.decoder_channels[0], self.decoder_channels[1]),
            *[UpBlockIN(self.decoder_channels[i], self.decoder_channels[i+1]) for i in range(1, len(self.decoder_channels) - 1)],
            nn.Conv2d(self.decoder_channels[-1], hyperparameters["channel_size"], 1, 1, 0)
        ) # Decoder to generate an image from edited/non-edited latent code

        style_channels = self.encoder_channels[-1]

        self.encoder_map_dim = torch.tensor([
            style_channels,
            hyperparameters["img_dim"] // 2**(len(self.encoder_channels) - 1),
            hyperparameters["img_dim"] // 2**(len(self.encoder_channels) - 1)
        ]) # Dimensions of encoder output

        self.enc_dim = torch.prod(self.encoder_map_dim)

        self.skip_mask = SkipNet(self.decoder_channels[-3]) # Attention-based skip connections, added in extended version

        self.direction_matrix = nn.Parameter(
            data=(1.0 if random_init else 0.001) * torch.randn(size=(len(self.tags), self.enc_dim)),
            requires_grad=True) # Matrix including learnable latent directions
        
        self.vectorization_unit = nn.ModuleList(
            [nn.Conv2d(self.enc_dim, self.latent_dim, 1, 1, 0), nn.Conv2d(self.latent_dim, self.enc_dim, 1, 1, 0)]
        ) # For the flexibility of adjusting latent dimensions

        self.mappers = nn.ModuleList(
            [ShiftMapper(len(self.tags[i]["attributes"])) for i in range(len(self.tags))]
        ) # Mapper parameters for latent-guided editing

        self.mse_loss = nn.MSELoss()
    
    """
    Encoding step for the generator
    :param x: Image tensor to be encoded
    """
    def encode(self, x):
        e = x
        s = None
        for layer_idx in range(len(self.encoder)):
            e = self.encoder[layer_idx](e)
            if layer_idx == 2:
                s = e
        e = self.vectorization_unit[0](e).reshape(e.shape[0], self.latent_dim)
        return e, s
    
    """
    Decoding a given enbedding e and residual features s
    :param e: Image embedding to be upsampled
    :param s: The residual features to be used to obtain attention mask
    """
    def decode(self, e, s):
        e = e.reshape(e.shape[0], self.latent_dim, 1, 1)
        x = self.vectorization_unit[1](e)
        x = x.reshape(e.shape[0], *self.encoder_map_dim)
        for layer_idx in range(len(self.decoder)):
            if layer_idx == (len(self.decoder) - 3):
                attn_mask = self.skip_mask(torch.cat((s,x), dim=1))
                x = attn_mask * s + (1 - attn_mask) * x
            x = self.decoder[layer_idx](x)
        return x

    """
    Extracting the feature scale given a tag and embedding
    :param e: The embedding which the extraction will be performed on
    :param tag: The index for the tag to perform extraction
    """
    def extract(self, e, tag):
        dir_vector = self.direction_matrix[tag, :]
        alpha = torch.mm(e, dir_vector.reshape(dir_vector.shape[0], 1)) / torch.dot(dir_vector, dir_vector)
        return alpha
    
    """
    Mapping a given z according to a feature scale
    :param z: The scalar value z to perform mapping on
    :param i: The tag to perform the mapping on
    :param j: The attribute to perform mapping on 
    (for the label 'with Bangs', the tag corresponds to 'Bangs' and the atrribute corresponds to 'with',
    check the configurations file for the indexes)
    """
    def map(self, z, i, j):
        return self.mappers[i](z, j)
    
    """
    Translates the given embedding e w.r.t alpha and tag
    :param e: The image embedding to be translated
    :param tag: The tag on which the translation will be performed on
    :param alpha: The scale for the translation
    """
    def translate(self, e, tag, alpha):
        dir_vector = torch.mul(self.direction_matrix[tag, :], alpha)
        return e + dir_vector
    
    """
    Loads generator weights from a checkpoint file
    :param ckpt_path: The full path to the checkpoint file
    """
    def load(self, ckpt_path):
        assert os.path.isfile(ckpt_path), "The given ckpt_path is not a file"
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict["gen_test"], strict=True)
        print(f"Successfully loaded model from {ckpt_path}!")

# Submodule Definitions
class SkipNet(nn.Module):
    def __init__(self, channel_in):
        super(SkipNet, self).__init__()
        # Note: convert to AdaIN with 2 blocks
        self.encoder = nn.Sequential(
            DownBlockIN(2 * channel_in, channel_in),
            DownBlockIN(channel_in, channel_in),
            DownBlockIN(channel_in, channel_in)
        )

        self.decoder = nn.Sequential(
            UpBlockIN(channel_in, channel_in),
            UpBlockIN(channel_in, channel_in),
            UpBlockIN(channel_in, channel_in)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat_1 = self.encoder[0](x)
        feat_2 = self.encoder[1](feat_1)
        feat_3 = self.encoder[2](feat_2)

        out = self.decoder[0](feat_3) + feat_2
        out = self.decoder[1](out) + feat_1
        out = self.sigmoid(self.decoder[2](out))

        return out
    

class ShiftMapper(nn.Module):
    def __init__(self, num_attributes):
        super(ShiftMapper, self).__init__()
        self.endpoints = nn.Parameter(data=torch.randn(num_attributes + 1), requires_grad=True)

    def forward(self, z, j):
        return z * (self.endpoints[j + 1] - self.endpoints[j]) + self.endpoints[j]


# Basic Blocks
# We use the block definitions provided in: https://github.com/imlixinyang/HiSD/tree/main
class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(F.avg_pool2d(self.conv1(self.activ(x.clone())), 2)))
        out = residual + out
        return out / math.sqrt(2)
    

class DownBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(in_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(self.in2(F.avg_pool2d(self.conv1(self.activ(self.in1(x.clone()))), 2))))
        out = residual + out
        return out / math.sqrt(2)
        

class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.conv1(F.interpolate(self.activ(x.clone()), scale_factor=2, mode='nearest'))))
        out = residual + out
        return out / math.sqrt(2)
    

class UpBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.in2(self.conv1(F.interpolate(self.activ(self.in1(x.clone())), scale_factor=2, mode='nearest')))))
        out = residual + out
        return out / math.sqrt(2)
    

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
    

    