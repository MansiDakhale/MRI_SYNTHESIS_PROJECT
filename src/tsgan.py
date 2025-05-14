import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Residual Block
# ------------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


# ------------------------------
# Attention Gate
# ------------------------------
class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Resize g to match the spatial dimensions of x
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  # Now dimensions should match
        psi = self.psi(psi)
        return x * psi


# ------------------------------
# Crop and Concatenate
# ------------------------------
def crop_and_concat(upsampled, bypass):
    # Ensure bypass is cropped to match upsampled size
    if bypass.size()[2:] != upsampled.size()[2:]:
        diffZ = bypass.size(2) - upsampled.size(2)
        diffY = bypass.size(3) - upsampled.size(3)
        diffX = bypass.size(4) - upsampled.size(4)
        
        # Handle both crop and pad cases
        if diffZ > 0 or diffY > 0 or diffX > 0:
            # Need to crop bypass
            bypass = bypass[:, :,
                max(0, diffZ // 2): min(bypass.size(2), diffZ // 2 + upsampled.size(2)),
                max(0, diffY // 2): min(bypass.size(3), diffY // 2 + upsampled.size(3)),
                max(0, diffX // 2): min(bypass.size(4), diffX // 2 + upsampled.size(4))
            ]
        else:
            # Need to pad bypass
            padZ = abs(diffZ)
            padY = abs(diffY)
            padX = abs(diffX)
            bypass = F.pad(bypass, (padX//2, padX-padX//2, padY//2, padY-padY//2, padZ//2, padZ-padZ//2))
    
    return torch.cat([upsampled, bypass], dim=1)


# ------------------------------
# U-Net 3D Generator
# ------------------------------
class UNet3DGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True),
                ResidualBlock3D(out_ch)
            )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.enc1 = down_block(2, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = down_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = down_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = down_block(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            down_block(256, 512),
            nn.Dropout3d(0.3)  # optional dropout
        )

        self.att4 = AttentionGate3D(512, 256, 128)
        self.up4 = up_block(512, 256)
        self.dec4 = down_block(512, 256)

        self.att3 = AttentionGate3D(256, 128, 64)
        self.up3 = up_block(256, 128)
        self.dec3 = down_block(256, 128)

        self.att2 = AttentionGate3D(128, 64, 32)
        self.up2 = up_block(128, 64)
        self.dec2 = down_block(128, 64)

        self.att1 = AttentionGate3D(64, 32, 16)
        self.up1 = up_block(64, 32)
        self.dec1 = down_block(64, 32)

        self.final = nn.Conv3d(32, 1, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, pret1, seg):
        if pret1.dim() == 4:
            pret1 = pret1.unsqueeze(1)
        if seg.dim() == 4:
            seg = seg.unsqueeze(1)

        x = torch.cat([pret1, seg], dim=1)

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Debug information
        # print(f"e1: {e1.shape}, e2: {e2.shape}, e3: {e3.shape}, e4: {e4.shape}, b: {b.shape}")

        # Decoder path with attention
        d4 = self.up4(b)
        g4 = self.att4(b, e4)  # Attention mechanism
        d4 = self.dec4(crop_and_concat(d4, g4))

        d3 = self.up3(d4)
        g3 = self.att3(d4, e3)
        d3 = self.dec3(crop_and_concat(d3, g3))

        d2 = self.up2(d3)
        g2 = self.att2(d3, e2)
        d2 = self.dec2(crop_and_concat(d2, g2))

        d1 = self.up1(d2)
        g1 = self.att1(d2, e1)
        d1 = self.dec1(crop_and_concat(d1, g1))

        return self.activation(self.final(d1))


# ------------------------------
# Patch-based Discriminator
# ------------------------------
class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(2, 32, 4, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)  # PatchGAN output
        )

    def forward(self, pret1, cet1):
        if pret1.dim() == 4:
            pret1 = pret1.unsqueeze(1)
        if cet1.dim() == 4:
            cet1 = cet1.unsqueeze(1)
        x = torch.cat([pret1, cet1], dim=1)
        return self.model(x)