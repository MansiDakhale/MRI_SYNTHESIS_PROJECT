import numpy as np
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class NormalizeToRange:
    def __call__(self, data):
        pre_t1, ce_t1 = data
        
        # Print shape for debugging
        print(f"Debug - pre_t1 shape: {pre_t1.shape}, ce_t1 shape: {ce_t1.shape}")
        
        # Handle different dimensionality cases
        if len(pre_t1.shape) > 4:
            # Case: tensor has more dimensions than expected
            # Assuming the first dimension might be batch size or an extra dimension
            pre_t1 = pre_t1.squeeze(0)  # Remove the first dimension
            ce_t1 = ce_t1.squeeze(0)
            
        if len(pre_t1.shape) < 4:
            # Case: tensor has fewer dimensions than expected
            # Add channel dimension if missing
            if len(pre_t1.shape) == 3:
                pre_t1 = pre_t1.unsqueeze(0)  # Add channel dimension
                ce_t1 = ce_t1.unsqueeze(0)
        
        # Now try to unpack - this should be safer
        try:
            c, d, h, w = pre_t1.shape
        except ValueError as e:
            print(f"Error unpacking shape {pre_t1.shape}. Attempting to reshape...")
            # Reshape as needed based on actual dimensionality
            if len(pre_t1.shape) == 3:
                # If it's 3D (d, h, w), add channel dimension
                pre_t1 = pre_t1.unsqueeze(0)
                ce_t1 = ce_t1.unsqueeze(0)
            elif len(pre_t1.shape) == 2:
                # If it's 2D (h, w), add channel and depth dimensions
                pre_t1 = pre_t1.unsqueeze(0).unsqueeze(0)
                ce_t1 = ce_t1.unsqueeze(0).unsqueeze(0)
            
            # Print new shapes
            print(f"Reshaped - pre_t1 shape: {pre_t1.shape}, ce_t1 shape: {ce_t1.shape}")
        
        # Normalize to range [0, 1]
        pre_t1_min, pre_t1_max = pre_t1.min(), pre_t1.max()
        ce_t1_min, ce_t1_max = ce_t1.min(), ce_t1.max()
        
        # Avoid division by zero
        pre_t1_range = pre_t1_max - pre_t1_min
        ce_t1_range = ce_t1_max - ce_t1_min
        
        if pre_t1_range == 0:
            pre_t1_norm = pre_t1 - pre_t1_min  # Will be all zeros
        else:
            pre_t1_norm = (pre_t1 - pre_t1_min) / pre_t1_range
        
        if ce_t1_range == 0:
            ce_t1_norm = ce_t1 - ce_t1_min  # Will be all zeros
        else:
            ce_t1_norm = (ce_t1 - ce_t1_min) / ce_t1_range
        
        return pre_t1_norm, ce_t1_norm


# Add other transform classes as needed...