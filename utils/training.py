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
import matplotlib.pyplot as plt
from . import imgs as img_utils

RESULTS_PATH = ".results/"
WEIGHTS_PATH = ".weights/"

hyper = get_hyperparams()
batch_size = hyper["batch_size"]
num_classes = hyper["num_classes"]
img_shape = hyper["image_shape"]
mode = hyper["mode"]
lr = hyper["learning_rate"]


def save_weights(model, epoch, loss, err, mode):
    weights_fname = "weights-%s-%d-%.3f-%.3f.pth" % (mode, epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save(
        {
            "startEpoch": epoch,
            "loss": loss,
            "error": err,
            "state_dict": model.state_dict(),
        },
        weights_fpath,
    )
    shutil.copyfile(weights_fpath, WEIGHTS_PATH + "latest.th")


def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights["startEpoch"]
    model.load_state_dict(weights["state_dict"])
    print(
        "loaded weights (lastEpoch {}, loss {}, error {})".format(
            startEpoch - 1, weights["loss"], weights["error"]
        )
    )
    return startEpoch


def get_predictions(output_batch):
    bs, c, h, w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_epistemic(outputs, predictive_mean, test_trials=20):
    result = torch.tensor(
        np.zeros((batch_size, img_shape[0], img_shape[1]), dtype=np.float32)
    ).cuda()
    target_sq = torch.einsum("bchw,bchw->bhw", [predictive_mean, predictive_mean]).data
    for i in range(test_trials):
        output_sq = torch.einsum(
            "bchw,bchw->bhw", [outputs[i], outputs[i]]
        ).data  # 질문하기
        result += output_sq - target_sq
    result /= test_trials
    return result


def error(preds, targets):
    assert preds.size() == targets.size()
    bs, h, w = preds.size()
    n_pixels = bs * h * w
    incorrect = float(preds.ne(targets).cpu().sum())
    err = incorrect / n_pixels
    return round(err, 5)


def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        if list(data[0].size())[0] != batch_size:
            break
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())
        optimizer.zero_grad()
        output = model(inputs)[0]  # [2, 12, 360, 480]
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        trn_loss += loss.data
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())
    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    torch.cuda.empty_cache()
    return float(trn_loss), trn_error


def train_aleatoric(model, trn_loader, optimizer, criterion, epoch):
    """Train aleatoric model"""
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        if list(data[0].size())[0] != batch_size:
            break
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())
        optimizer.zero_grad()
        output, logvar = model(inputs)  # tuple of [output, log_var]
        loss = criterion((output, logvar), targets)
        loss.backward()
        optimizer.step()
        trn_loss += loss.data
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())
    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    torch.cuda.empty_cache()
    return float(trn_loss), trn_error


def test(model, test_loader, criterion, epoch=1):
    """Baseline Test
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.eval()  # dropout off
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        if list(data.size())[0] != batch_size:
            break
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        output = model(data)[0]
        test_loss += criterion(output, target).data
        pred = get_predictions(output)
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def test_aleatoric(model, test_loader, criterion, epoch=1):
    """Baseline Test
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.eval()  # dropout off
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        if list(data.size())[0] != batch_size:
            break
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).data
        pred = get_predictions(output[0])
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def test_epistemic(model, test_loader, criterion, test_trials=20, epoch=1):
    """Epistemic model Test
    Please turn on Dropout!
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.train()  # train mode: turn on dropout
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        print(torch.cuda.memory_allocated(device=0))
        if list(data.size())[0] != batch_size:
            break
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        outputs = model(data)[0].data
        for i in range(test_trials - 1):
            outputs += model(data)[0].data
        output = outputs / test_trials  # mean
        pred = get_predictions(output)
        test_loss += criterion(output, target).data
        test_error += error(pred, target.data.cpu())
        torch.cuda.empty_cache()
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def test_combined(model, test_loader, criterion, test_trials=20, epoch=1):
    """Combined model Test
    Please turn on Dropout!
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.train()  # train mode: turn on dropout
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        if list(data.size())[0] != batch_size:
            break
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        outputs, log_var = model(data)
        outputs = outputs.data
        for i in range(test_trials - 1):
            outputs += model(data)[0].data
        output = outputs / test_trials
        pred = get_predictions(output)
        test_loss += criterion((output, log_var), target).data
        test_error += error(pred, target.data.cpu())
        torch.cuda.empty_cache()
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input, target, pred])
    return predictions


def view_sample_predictions(model, loader, n):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    output = model(data)[0]
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])


def view_sample_predictions_with_uncertainty(
    model, inputs, targets, n, test_trials=100
):
    model.train()
    data = Variable(inputs.cuda(), volatile=True).view(1, 3, img_shape[0], img_shape[1])
    label = Variable(targets.cuda())
    output, log_var = model(data)
    shape = (1, 1, num_classes, img_shape[0], img_shape[1])
    outputs = model(data)[0].view(shape).data
    for i in range(test_trials - 1):
        output = model(data)[0].view(shape).data
        outputs = torch.cat([outputs, output], dim=0)
    predictive_mean = outputs.mean(dim=0)  # mean
    pred = get_predictions(predictive_mean)[0]
    base_path = "./combined/"
    # uncertainty
    epistemic = get_epistemic(outputs, predictive_mean, test_trials)  # check shape
    aleatoric = log_var[0]

    img_utils.view_image(inputs, path=base_path, n=n, mode="input")
    img_utils.view_annotated(targets, path=base_path, n=n, mode="target")
    img_utils.view_annotated(pred, path=base_path, n=n, mode="pred")
    img_utils.view_image_with_uncertainty(
        inputs, epistemic, path=base_path, n=n, mode="epistemic"
    )
    img_utils.view_image_with_uncertainty(
        inputs, aleatoric, path=base_path, n=n, mode="aleatoric"
    )


def save_result(
    train_loss: float, train_err: float, val_loss: float, val_error: float, epoch: int
) -> None:
    save = hyper
    save["train_loss"] = train_loss
    save["train_err"] = train_err
    save["val_loss"] = val_loss
    save["val_error"] = val_error

    save_ = sorted(save.items(), key=(lambda x: x[0]))
    dataframe = pd.DataFrame(save_)
    dataframe.to_csv(
        "./.results/{}-lr-{}-epoch-{}.csv".format(mode, lr, epoch), encoding="utf-8"
    )
