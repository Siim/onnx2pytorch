import torch
from torch import nn

class GatherElements(nn.Module):
    def forward(self, data, indices, axis=0):
        # We must roll axis to first dim which is what PyTorch's gather expects
        data = data.roll(-axis, range(data.dim()))
        indices = indices.roll(-axis, range(indices.dim()))

        # Use PyTorch's gather operation with dim=0
        gathered_elements = data.gather(0, indices)

        # Return axes to their original order
        gathered_elements = gathered_elements.roll(axis, range(gathered_elements.dim()))
        return gathered_elements
