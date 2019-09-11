import os
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx as torch_onnx

from torchvision import datasets, transforms
from torch.autograd import Variable

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

from src import *

def load_model(model_path):
    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Load model:', model_path)
    model.eval()

    return model

def main():
    model_dir = '/data/data/crowd-counting/models/pth'
    model_name = '12crop_mean_model_best.pth'
    # model_name = '12crop_ownmean_model_best.pth'
    # model_name = '12resize_mean_model_best.pth'
    # model_name = '12resize_ownmean_model_best.pth'
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)

    model_onnx_dir = '/data/data/crowd-counting/models/onnx'
    model_onnx_path = os.path.join(model_onnx_dir, model_name.replace('.pth', '.onnx'))

    input_shape = (3, 1080, 1920)
    dummy_input = Variable(torch.randn(1, *input_shape)).cuda()

    # generate onnx
    torch_onnx.export(model, dummy_input, model_onnx_path, export_params=True, verbose=True)
    print("Export of torch_model.onnx complete!")

if __name__ == '__main__':
    main()