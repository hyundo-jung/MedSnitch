import torch
from torch import nn, sigmoid
import torch.optim as optim
import pandas as pd
import xgboost
from xgboost import XGBClassifier
import os
import numpy as np
import joblib

# Neural Network Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)
