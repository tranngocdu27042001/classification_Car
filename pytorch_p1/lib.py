import torch
import torchvision
from torchvision import models, transforms
import numpy as np
from PIL import Image
import torch.utils.data as data
import glob
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch.random
import numpy.random
import torch.optim as optim
from tqdm import tqdm

root_path = 'C:\\Users\\Mr Du\\PycharmProjects\\pytorch_project\\data\\'

resize = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

batch_size=4
epochs=2
