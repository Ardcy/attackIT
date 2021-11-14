import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import utils
import Utils
from net import Net, Vgg16
from Utils import *
from option import Options

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import PGD


import torchvision.datasets as dsets

def end_batch(batch):
    
    (b, g, r) = torch.chunk(batch, 3)
    batch = torch.cat((r, g, b))
    #batch = batch.transpose(0, 1)
    return batch

def imshow(img, title):
    npimg = img.numpy()
    npimg = npimg.astype(int)
    
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))

ci = []
si = []
o = []
def mxnet(content_image_path,style_image_path):

    n_batch = 1
    content_image = Utils.tensor_load_rgbimage(content_image_path, size=1024, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style = Utils.tensor_load_rgbimage(style_image_path, size=512)
    style = style.unsqueeze(0)
    style = Variable(Utils.preprocess_batch(style),requires_grad = True)

    mse_loss = torch.nn.MSELoss()

    style_model = Net(ngf=128)
    model_dict = torch.load('models/21styles.model')
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    style_v = Variable(style,requires_grad=True)
    content_image = Variable(Utils.preprocess_batch(content_image),requires_grad = True)
    style_model.setTarget(style_v)

    vgg = Vgg16()
    save_model = models.vgg16(pretrained=True).state_dict()
    model_dict =  vgg.state_dict()
    state_dict = {k:list(save_model.values())[idx] for idx,k in enumerate(model_dict.keys())}
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    vgg.load_state_dict(model_dict)


    style_v1 = Utils.subtract_imagenet_mean_batch(style_v.cpu())
    features_style = vgg(style_v1)
    gram_style = [Utils.gram_matrix(y) for y in features_style]

    output = style_model(content_image)
    xc = Variable(content_image.data.clone())

    output = Utils.subtract_imagenet_mean_batch(output)
    xc = Utils.subtract_imagenet_mean_batch(xc)

    features_y = vgg(output)
    features_xc = vgg(xc)

    f_xc_c = Variable(features_xc[1].data, requires_grad=False)
    content_loss = 1.0 * mse_loss(features_y[1], f_xc_c)

    style_loss = 0.
    for m in range(len(features_y)):
        gram_y = Utils.gram_matrix(features_y[m])
        gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(1, 1, 1)
        style_loss += 5.0 * mse_loss(gram_y, gram_s[:n_batch, :, :])

    total_loss = content_loss + style_loss
    print("total_loss = {}".format(total_loss))
    #total_loss.backward()

    Utils.tensor_save_bgrimage(output.data[0], "output.jpg", 1)

    global ci,si,o
    ci = content_image.clone()
    si = style_v.clone()
    o = output.clone()
    print(type(ci))

content_image_path = "images/content/venice-boat.jpg"
style_image_path = "images/9styles/feathers.jpg"
mxnet(content_image_path,style_image_path)

x = ci.detach()
imshow(end_batch(x[0]),'content')

x = si.detach()
imshow(end_batch(x[0]),'style')

x = o.detach()
imshow(end_batch(x[0]),'result')