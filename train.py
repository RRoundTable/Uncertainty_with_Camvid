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
from utils.training import save_result
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

# utils.imgs.view_image(inputs[0])
# utils.imgs.view_annotated(targets[0])

_criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda(), reduction="none").cuda()


def custom_cirterion(y_pred, y_true):
    """Aleatoric loss function
    See paper at 2.2 Heteroscedastic Aleatoric Uncertainty (5)
    """
    output, log_var = y_pred
    log_var = log_var[0]
    loss = torch.exp(-1 * log_var) * 0.5 + _criterion(output, y_true) + 0.5 * log_var
    return loss.sum()


# hyperparameters
LR = hyper["learning_rate"]
LR_DECAY = hyper["lr_decay"]
DECAY_EVERY_N_EPOCHS = hyper["decay_per_n_epoch"]
N_EPOCHS = hyper["n_epoch"]

torch.cuda.manual_seed(0)

if mode == "base":
    model = tiramisu.FCDenseNet57(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda()).cuda()
    test = train_utils.test
    train = train_utils.train

elif mode == "epistemic":
    model = tiramisu.FCDenseNet57(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda()).cuda()
    test = train_utils.test_epistemic
    train = train_utils.train

elif mode == "aleatoric":
    model = tiramisu.FCDenseNet57_aleatoric(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = custom_cirterion
    test = train_utils.test_aleatoric
    train = train_utils.train_aleatoric

elif mode == "combined":
    model = tiramisu.FCDenseNet57_aleatoric(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = custom_cirterion
    test = train_utils.test_combined
    train = train_utils.train_aleatoric

if __name__ == "__main__":
    val_tmp = sys.maxsize
    print("Mode: {}".format(mode))
    for epoch in range(1, N_EPOCHS + 1):
        since = time.time()
        # Train
        trn_loss, trn_err = train(model, train_loader, optimizer, criterion, epoch)
        print(
            "Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}".format(
                epoch, trn_loss, 1 - trn_err
            )
        )

        time_elapsed = time.time() - since
        print(
            "Train Time {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )

        # Test
        val_loss, val_err = test(model, val_loader, criterion, epoch)
        print("Val - Loss: {:.4f} | Acc: {:.4f}".format(val_loss, 1 - val_err))
        time_elapsed = time.time() - since
        print(
            "Total Time {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60)
        )

        if val_tmp < val_loss:  # early stopping
            break
        else:
            val_tmp = val_loss
        # save results
        save_result(trn_loss, trn_err, val_loss, val_err, epoch)

        ### Adjust Lr ###
        train_utils.adjust_learning_rate(
            LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS
        )
    ### Checkpoint ###
    train_utils.save_weights(model, epoch, val_loss, val_err, mode=mode)
