import os
import sys
import math
import string
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from hyperparams import get_hyperparams
import torch.nn.functional as F
import pandas as pd

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

hyper = get_hyperparams()
batch_size = hyper['batch_size']
num_classes = hyper['num_classes']
img_shape = hyper['image_shape']
mode = hyper['mode']
lr = hyper['learning_rate']

def get_epistemic_uncertainty(model, dataloader, test_trials):
    model.train()
    for data, target in dataloader:
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda(), volatile=True) # (batch_size, num_classes, height, width)
        result = torch.tensor(np.zeros(batch_size, img_shape[0], img_shape[1]))
        for i in range(test_trials):
            output = model(data)[0]
            output_sq = torch.einsum('bchw,bchw->bhw', [output, output]) # 질문하기
            target_sq = torch.einsum('bchw,bchw->bhw', [target, target]) # 확인하기
            result += output_sq - target_sq
        result /= test_trials
    return result

def get_epistemic(output, predictive_mean, test_trials=20):
    result = torch.tensor(np.zeros(batch_size, img_shape[0], img_shape[1]))
    for i in range(test_trials):
        output_sq = torch.einsum('bchw,bchw->bhw', [output, output])  # 질문하기
        target_sq = torch.einsum('bchw,bchw->bhw', [predictive_mean, predictive_mean])  # 확인하기
        result += output_sq - target_sq
    result /= test_trials
    return result





def get():
    return None