import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def create_writer(log_dir="runs/3d_mri_synthesis"):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)

def log_losses(writer, g_loss, d_loss, step):
    writer.add_scalar("Loss/Generator", g_loss, step)
    writer.add_scalar("Loss/Discriminator", d_loss, step)

def log_mri_slices(writer, real, fake, step, slice_index=None):
    """
    Logs center (or custom) slice of real and fake CeT1 MRI volumes to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        real: Ground truth CeT1 tensor [B, 1, D, H, W]
        fake: Generated CeT1 tensor [B, 1, D, H, W]
        step: Current training step
        slice_index: Custom slice to visualize (defaults to center slice)
    """
    if slice_index is None:
        slice_index = real.shape[2] // 2  # Center slice along depth

    # Extract 2D slices from center of 3D volumes
    real_slice = real[:, :, slice_index, :, :]  # [B, 1, H, W]
    fake_slice = fake[:, :, slice_index, :, :]  # [B, 1, H, W]

    # Normalize for display
    real_grid = make_grid(real_slice, nrow=4, normalize=True, scale_each=True)
    fake_grid = make_grid(fake_slice, nrow=4, normalize=True, scale_each=True)

    # Add individual visualizations
    writer.add_image("MRI/Real_CeT1", real_grid, step)
    writer.add_image("MRI/Fake_CeT1", fake_grid, step)

    # Optional: Side-by-side visualization
    comparison = torch.cat([real_slice, fake_slice], dim=3)  # Concatenate along width
    comparison_grid = make_grid(comparison, nrow=4, normalize=True, scale_each=True)
    writer.add_image("MRI/Real_vs_Fake", comparison_grid, step)
