import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
from baseline import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils
from utils.training import load_weights
from hyperparams import get_hyperparams

sys.setrecursionlimit(15000)
hyper = get_hyperparams()
dropout = hyper["dropout"]
mode = hyper["mode"]

assert mode in ["base", "epistemic", "aleatoric", "combined"], "Wrong mode!"

# CAMVID_PATH = Path('./CamVid/Camvid/')
CAMVID_PATH = "./CamVid/CamVid/"
RESULTS_PATH = Path(".results/")
WEIGHTS_PATH = Path(".weights/")
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = hyper["batch_size"]

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose(
    [
        # joint_transforms.JointRandomCrop(224), # commented for fine-tuning
        joint_transforms.JointRandomHorizontalFlip()
    ]
)

train_dset = camvid.CamVid(
    CAMVID_PATH,
    "train",
    joint_transform=train_joint_transformer,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)

train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size, shuffle=True
)

val_dset = camvid.CamVid(
    CAMVID_PATH,
    "val",
    joint_transform=None,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)

test_dset = camvid.CamVid(
    CAMVID_PATH,
    "test",
    joint_transform=None,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False
)

print("Train: %d" % len(train_loader.dataset.imgs))
print("Val: %d" % len(val_loader.dataset.imgs))
print("Test: %d" % len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

# hyperparameters
LR = hyper["learning_rate"]
LR_DECAY = hyper["lr_decay"]
DECAY_EVERY_N_EPOCHS = hyper["decay_per_n_epoch"]
N_EPOCHS = hyper["n_epoch"]

# load combined model
model_path = "./.weights/weights-combined-11-27987.752-0.343.pth"
model = tiramisu.FCDenseNet57_aleatoric(n_classes=12, dropout=dropout).cuda()
load_weights(model, model_path)

if __name__ == "__main__":
    print("Visualizing ...")
    # Make visualization
    i = 0
    for inputs, targets in train_dset:
        print("{} - th".format(i))
        train_utils.view_sample_predictions_with_uncertainty(model, inputs, targets, i)
        i += 1
        if i > 100:
            break
