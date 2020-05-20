#!/usr/bin/python
import os
from skimage.util.shape import view_as_windows
import torch
import torch.nn as nn
import numpy as np
from transformlayer import ColorJitterLayer


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def random_crop(imgs, out= 84): # Changed the input and output shape to match the numpy shapes
    '''
    args:
    imgs: np array shape (B,H,W,C)
    out: output size (e.g. 84)
    return np.array (B,H,W,C)
    '''
    n, h, w,c = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, out, out, c), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[h11:h11 + out, w11:w11 + out,:]
    return cropped

'''
def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)  -  Changed this to batch images with shape (B,H,W,C)
        output shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[1] # Nawid - Changed this to get image size
    crop_max = img_size - output_size
    #imgs = np.transpose(imgs, (0, 2, 3, 1)) #  Nawid - No longer required as I am using a different input size
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs # Nawid -Output shape is the required pytorch shape (channels in dim 1) (B,C,H,W)
'''

def center_crop_image(image, output_size):
    '''
    args:
    imgs, batch with shape (B,H,W,C)
    output shape (B,H,W,C)
    '''
    h, w = image.shape[1:-1] #  Obtain the height and width of a 4D tensor
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w,:]
    #image = image.reshape((image.shape[0], image.shape[3], image.shape[1], image.shape[2]))
    return image


def random_color_jitter(imgs,batch_size=128,frames=4):
    """
        inputs np array outputs tensor
    """
    b,c,h,w = imgs.shape
    imgs = imgs.view(-1,3,h,w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                                hue=0.5,
                                                p=1.0,
                                                batch_size=batch_size,
                                                stack_size = frames))

    imgs = transform_module(imgs).view(b,c,h,w)
    return imgs


def soft_update_params(net, target_net, tau): # Nawid-update for the momentum encoder
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear): # Nawid-  Weight init for linear layers
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d): # Nawid- weight inti for conv
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
