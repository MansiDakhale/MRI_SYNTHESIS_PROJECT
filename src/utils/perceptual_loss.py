import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features for 2D slices from 3D MRI volumes.
    This works by extracting features from the middle slice of each volume.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Use the proper weights parameter instead of pretrained
        vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]  # Use layers up to relu_2_2
        
        self.vgg = vgg16.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()
        
        # Register mean and std for input normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        """
        Args:
            x: Input tensor of shape [B, 1, H, W] or [B, 1, D, H, W]
            y: Target tensor of shape [B, 1, H, W] or [B, 1, D, H, W]
        
        Returns:
            Perceptual loss between x and y
        """
        # Handle 3D volumes by using a middle slice
        if x.dim() == 5:  # If input is [B, C, D, H, W]
            # Take middle slice along depth dimension
            d_idx = x.shape[2] // 2
            x = x[:, :, d_idx]  # Now [B, C, H, W]
            y = y[:, :, d_idx]  # Now [B, C, H, W]
        
        # Convert to 3-channel if input is single channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # Normalize with ImageNet mean and std as VGG expects
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # Resize inputs to 224x224 which is what VGG expects
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = torch.nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
            
        x_feat = self.vgg(x)
        y_feat = self.vgg(y)

        return self.criterion(x_feat, y_feat)