import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.linear = nn.Linear(256, 256)

    def forward(self, batch):
        x = torch.rand(2, 256)
        output = self.linear(x)

        return (
            output
        )
