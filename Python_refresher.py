import torch
import torch.nn as nn
import numpy
import os

with torch.no_grad():
    preds = model(x)