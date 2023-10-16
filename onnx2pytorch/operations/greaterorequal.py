import torch
from torch import nn

class GreaterOrEqual(nn.Module):
    def forward(self, tensor1, tensor2):
        return tensor1 >= tensor2
