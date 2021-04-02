import os
import gym
import time
import torch
import random
import warnings
import pybullet_envs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym import wrappers
from torch.autograd import Variable
from collections import deque
warnings.filterwarnings("ignore")

is_gpu = torch.cuda.is_available()
if is_gpu:
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
