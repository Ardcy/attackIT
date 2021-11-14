import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import *

from models import Holdout, Target
from utils import imshow




batch_size = 24

cifar10_train = dsets.CIFAR10(root='./data', train=True,
                              download=True, transform=transforms.ToTensor())
cifar10_test  = dsets.CIFAR10(root='./data', train=False,
                              download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(cifar10_train,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_test,
                                          batch_size=batch_size,
                                          shuffle=False)

images, labels = iter(train_loader).next()
imshow(torchvision.utils.make_grid(images, normalize=True), "Train Image")

model = Holdout()
model.load_state_dict(torch.load("./checkpoint/holdout.pth"))
model = model.eval().cuda()



atks = [
    DeepFool(model, steps=100),
    FGSM(model, eps=8/255),
    BIM(model, eps=8/255, alpha=2/255, steps=100),
    PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True)
]

atks_name = [
    "DeepFool",
    "FGSM",
    "BIM",
    "PGD",
]

for idx,atk in enumerate(atks):
    atk.set_return_type('int') # Save as integer.
    atk.save(data_loader=test_loader, save_path="./data/cifar10.pt", verbose=True)

    adv_images, adv_labels = torch.load("./data/cifar10.pt")
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

    model = Target().cuda()
    model.load_state_dict(torch.load("./checkpoint/target.pth"))

    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:
        
        images = images.cuda()
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    print(atks_name[idx])    
    print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

    model.eval()

    correct = 0
    total = 0

    for images, labels in adv_loader:
        
        images = images.cuda()
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    
    

    print(atks_name[idx])
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))