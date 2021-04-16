import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.truth = torch.rand(2, 256)

    def forward(self, pred, truth):
        total_loss = self.mse_loss(pred, self.truth)
        mel_loss = self.mse_loss(pred, self.truth)
        postnet_mel_loss = self.mse_loss(pred, self.truth)
        energy_loss = self.mse_loss(pred, self.truth)
        f0_loss = self.mse_loss(pred, self.truth)
        duration_loss = self.mse_loss(pred, self.truth)

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            energy_loss,
            f0_loss,
            duration_loss,
        )
