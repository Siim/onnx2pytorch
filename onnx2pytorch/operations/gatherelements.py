import torch
from torch import nn

class GatherElements(nn.Module):
    def forward(self, data, indices, axis=0):
        # Ensure axis is a valid dimension
        if not (0 <= axis < data.dim()):
            raise ValueError(("Expected axis to be an integer between 0 and the number "
                              "of dimensions of input tensor, but got axis {} "
                              "for input of dimension {}").format(axis, data.dim()))

        # Ensure data and indices have the same dimensions
        if data.dim() != indices.dim():
            raise ValueError(("Expected data and indices to have the same number of "
                              "dimensions, but got {} and {}, respectively.")
                             .format(data.dim(), indices.dim()))

        # Roll the axis to be gathered to the front of the tensor
        data = data.roll(-axis, range(data.dim()))
        indices = indices.roll(-axis, range(indices.dim()))

        # Use PyTorch's gather operation with dim=0
        gathered_elements = data.gather(0, indices)

        # Return axes to their original order
        gathered_elements = gathered_elements.roll(axis, range(gathered_elements.dim()))
        
        return gathered_elements
