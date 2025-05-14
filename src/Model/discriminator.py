import torch
import torch.nn as nn

class TSGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[32, 64, 128, 256]):
        super(TSGANDiscriminator, self).__init__()
        layers = []
        for feature in features:
            layers.append(nn.Conv3d(in_channels, feature, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm3d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = feature
        layers.append(nn.Conv3d(features[-1], 1, kernel_size=4, padding=1))  # PatchGAN
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
