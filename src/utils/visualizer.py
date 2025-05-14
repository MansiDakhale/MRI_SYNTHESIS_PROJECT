import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

def visualize_mri_comparison(pre_t1, ce_t1_pred, ce_t1_gt=None, output_path=None, slice_idx=None, dice_score=None):
    """
    Visualize comparison between pre-contrast, predicted contrast-enhanced, and ground truth contrast-enhanced MRIs.
    
    Args:
        pre_t1 (np.ndarray): Pre-contrast T1 volume
        ce_t1_pred (np.ndarray): Predicted contrast-enhanced T1 volume
        ce_t1_gt (np.ndarray, optional): Ground truth contrast-enhanced T1 volume
        output_path (str, optional): Path to save the visualization
        slice_idx (tuple, optional): Custom slice indices for each dimension (D, H, W)
        dice_score (float, optional): Dice score to display
    """
    # Determine if we have ground truth
    has_gt = ce_t1_gt is not None
    
    # Default to middle slices if not specified
    if slice_idx is None:
        d = pre_t1.shape[0] // 2
        h = pre_t1.shape[1] // 2
        w = pre_t1.shape[2] // 2
    else:
        d, h, w = slice_idx
    
    # Create a figure with 3 rows (axial, sagittal, coronal)
    # and 2 or 3 columns depending on ground truth availability
    fig, axes = plt.subplots(3, 3 if has_gt else 2, figsize=(12, 12))
    
    # Common colormap and normalization
    cmap = 'gray'
    norm = Normalize(vmin=0, vmax=1)
    
    # Row 1: Axial view (top-down)
    axes[0, 0].imshow(pre_t1[d, :, :], cmap=cmap, norm=norm)
    axes[0, 0].set_title('Pre-contrast T1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ce_t1_pred[d, :, :], cmap=cmap, norm=norm)
    axes[0, 1].set_title('Predicted CE-T1')
    axes[0, 1].axis('off')
    
    if has_gt:
        axes[0, 2].imshow(ce_t1_gt[d, :, :], cmap=cmap, norm=norm)
        axes[0, 2].set_title('Ground Truth CE-T1')
        axes[0, 2].axis('off')
    
    # Row 2: Sagittal view (side)
    axes[1, 0].imshow(pre_t1[:, h, :], cmap=cmap, norm=norm)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ce_t1_pred[:, h, :], cmap=cmap, norm=norm)
    axes[1, 1].axis('off')
    
    if has_gt:
        axes[1, 2].imshow(ce_t1_gt[:, h, :], cmap=cmap, norm=norm)
        axes[1, 2].axis('off')
    
    # Row 3: Coronal view (front)
    axes[2, 0].imshow(pre_t1[:, :, w], cmap=cmap, norm=norm)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(ce_t1_pred[:, :, w], cmap=cmap, norm=norm)
    axes[2, 1].axis('off')
    
    if has_gt:
        axes[2, 2].imshow(ce_t1_gt[:, :, w], cmap=cmap, norm=norm)
        axes[2, 2].axis('off')
    
    # Add labels
    axes[0, 0].set_ylabel('Axial', fontsize=14)
    axes[1, 0].set_ylabel('Sagittal', fontsize=14)
    axes[2, 0].set_ylabel('Coronal', fontsize=14)
    
    if dice_score is not None:
        plt.suptitle(f'MRI Comparison (Dice Score: {dice_score:.4f})', fontsize=16)
    else:
        plt.suptitle('MRI Comparison', fontsize=16)
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def visualize_training_progress(pre_t1, ce_t1_gt, ce_t1_pred, epoch, step, output_dir):
    """
    Visualize training progress by showing pre-contrast, ground truth, and predicted contrast-enhanced MRIs.
    
    Args:
        pre_t1 (torch.Tensor): Pre-contrast T1 volume [1, 1, D, H, W]
        ce_t1_gt (torch.Tensor): Ground truth contrast-enhanced T1 volume [1, 1, D, H, W]
        ce_t1_pred (torch.Tensor): Predicted contrast-enhanced T1 volume [1, 1, D, H, W]
        epoch (int): Current epoch
        step (int): Current step
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy and get first sample
    pre_t1 = pre_t1[0, 0].cpu().numpy()
    ce_t1_gt = ce_t1_gt[0, 0].cpu().numpy()
    ce_t1_pred = ce_t1_pred[0, 0].detach().cpu().numpy()
    
    # Visualization file path
    output_path = os.path.join(output_dir, f'epoch_{epoch}_step_{step}.png')
    
    # Create visualization
    visualize_mri_comparison(pre_t1, ce_t1_pred, ce_t1_gt, output_path)

def create_difference_map(ce_t1_pred, ce_t1_gt, output_path=None):
    """
    Create a difference map between predicted and ground truth contrast-enhanced MRIs.
    
    Args:
        ce_t1_pred (np.ndarray): Predicted contrast-enhanced T1 volume
        ce_t1_gt (np.ndarray): Ground truth contrast-enhanced T1 volume
        output_path (str, optional): Path to save the visualization
    """
    # Calculate absolute difference
    diff = np.abs(ce_t1_pred - ce_t1_gt)
    
    # Get middle slices
    d = diff.shape[0