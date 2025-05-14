#Claude code
# 2nd Model

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()  # Simplified super() call
        # Use nn.ModuleList for systematic access to encoder/decoder blocks
        self.encoders = nn.ModuleList([
            self._make_encoder_block(in_channels, 16),
            self._make_encoder_block(16, 32)
        ])
        
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(2)])
        
        self.bottleneck = self._make_conv_block(32, 64)
        
        # Store upsampling layers in ModuleList
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.ConvTranspose3d(32, 16, 2, stride=2)
        ])
        
        self.decoders = nn.ModuleList([
            self._make_decoder_block(64, 32),
            self._make_decoder_block(32, 16)
        ])
        
        # Separate output layer with optional normalization
        self.out = nn.Sequential(
            nn.Conv3d(16, out_channels, 1),
            nn.Sigmoid()
        )
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with inplace ReLU for better memory efficiency"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),  # Remove bias when using BatchNorm
            nn.ReLU(inplace=True),  # Use inplace ReLU to save memory
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def _make_encoder_block(self, in_channels, out_channels):
        """Create an encoder block"""
        return self._make_conv_block(in_channels, out_channels)
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block"""
        return self._make_conv_block(in_channels, out_channels)
    
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for i, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            x = encoder(x)
            encoder_outputs.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            # Use encoder outputs in reverse order for skip connections
            skip_connection = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip_connection], dim=1)
            x = decoder(x)
            
        return self.out(x)