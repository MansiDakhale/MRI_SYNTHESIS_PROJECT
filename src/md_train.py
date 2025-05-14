import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import argparse
import numpy as np
import random

from utils.losses import dice_loss
from utils.perceptual_loss import PerceptualLoss
from utils.early_stopping import EarlyStopping
from utils.logger import create_writer, log_losses, log_mri_slices
from tsgan import UNet3DGenerator, Discriminator3D
from torch.utils.data import DataLoader
#from transform import get_train_transforms, get_val_transforms
from dataset import MRISegmentationDataset as MRIDataset

# Function to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Argument parser for command line options
def parse_args():
    parser = argparse.ArgumentParser(description='MRI Synthesis Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='runs/3d_mri_synthesis', help='Directory for tensorboard logs')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='Weight for perceptual loss')
    parser.add_argument('--lambda_adv', type=float, default=0.01, help='Weight for adversarial loss')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    generator = UNet3DGenerator().to(device)
    discriminator = Discriminator3D().to(device)
    
    # Initialize loss functions
    criterion_dice = dice_loss()
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_adv = nn.BCEWithLogitsLoss()
    
    # Initialize optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Initialize schedulers for learning rate decay
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5)
    
    # Initialize early stopping
    save_path = os.path.join(args.save_dir, 'best_generator.pt')
    early_stopper = EarlyStopping(patience=args.patience, verbose=True, path=save_path)
    
    # Initialize tensorboard writer
    writer = create_writer(log_dir=args.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Load datasets
    print("Loading datasets...")
    #train_transforms = get_train_transforms()
    #val_transforms = get_val_transforms()
    
    train_dir = r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\Resized_dataset\train_resized"
    val_dir = r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\Resized_dataset\val_resized"
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Warning: Train directory {train_dir} does not exist!")
    if not os.path.exists(val_dir):
        print(f"Warning: Validation directory {val_dir} does not exist!")
    
    try:
        train_dataset = MRIDataset(data_dir=train_dir)
        val_dataset = MRIDataset(data_dir=val_dir)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Checking directory structure and file availability...")
        
        # Basic directory structure check
        if os.path.exists(train_dir):
            print(f"Train directory exists: {train_dir}")
            print(f"Contents: {os.listdir(train_dir)}")
        if os.path.exists(val_dir):
            print(f"Validation directory exists: {val_dir}")
            print(f"Contents: {os.listdir(val_dir)}")
            
        raise
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_dice_loss = 0
        epoch_perceptual_loss = 0
        epoch_adv_loss = 0
        
        for pre_t1, ce_t1 in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{args.epochs}]"):
            pre_t1, ce_t1 = pre_t1.to(device), ce_t1.to(device)
            batch_size = pre_t1.size(0)
            
            # --------- Train Discriminator --------- #
            optimizer_d.zero_grad()
            
            # Generate fake contrast-enhanced images
            with torch.no_grad():
                fake_ce = generator(pre_t1)
            
            # Real and fake discriminator outputs
            real_logits = discriminator(ce_t1)
            fake_logits = discriminator(fake_ce.detach())
            
            # Labels for adversarial loss
            real_labels = torch.ones(batch_size, 1, 4, 4, 4).to(device)  # Adjust size based on your discriminator output
            fake_labels = torch.zeros(batch_size, 1, 4, 4, 4).to(device)
            
            # Discriminator losses
            loss_d_real = criterion_adv(real_logits, real_labels)
            loss_d_fake = criterion_adv(fake_logits, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            loss_d.backward()
            optimizer_d.step()
            
            # --------- Train Generator --------- #
            optimizer_g.zero_grad()
            
            # Generate fake images again (since we've updated D)
            fake_ce = generator(pre_t1)
            
            # Compute individual loss components
            fake_logits = discriminator(fake_ce)
            
            dice_loss = criterion_dice(fake_ce, ce_t1)
            
            # For perceptual loss, we use the middle slice
            perceptual_loss = criterion_perceptual(fake_ce, ce_t1)
            
            adv_loss = criterion_adv(fake_logits, real_labels)
            
            # Combine losses with weighting factors
            loss_g = dice_loss + args.lambda_perceptual * perceptual_loss + args.lambda_adv * adv_loss
            
            loss_g.backward()
            optimizer_g.step()
            
            # Update running losses
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            epoch_dice_loss += dice_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()
            epoch_adv_loss += adv_loss.item()
            
            # --------- Logging --------- #
            if global_step % 10 == 0:
                log_losses(writer, loss_g.item(), loss_d.item(), global_step)
                
                # Log individual loss components
                writer.add_scalar('Train/Dice_Loss', dice_loss.item(), global_step)
                writer.add_scalar('Train/Perceptual_Loss', perceptual_loss.item(), global_step)
                writer.add_scalar('Train/Adversarial_Loss', adv_loss.item(), global_step)
            
            if global_step % 50 == 0:
                log_mri_slices(writer, ce_t1, fake_ce, global_step)
            
            global_step += 1
        
        # Calculate average losses for epoch
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_dice_loss = epoch_dice_loss / len(train_loader)
        avg_perceptual_loss = epoch_perceptual_loss / len(train_loader)
        avg_adv_loss = epoch_adv_loss / len(train_loader)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"[Epoch {epoch+1}] Time: {epoch_duration:.2f}s | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
        print(f"    Dice: {avg_dice_loss:.4f} | Perceptual: {avg_perceptual_loss:.4f} | Adversarial: {avg_adv_loss:.4f}")
        
        # --------- Validation --------- #
        generator.eval()
        val_loss = 0
        val_dice_loss = 0
        
        with torch.no_grad():
            for pre_t1, ce_t1 in val_loader:
                pre_t1, ce_t1 = pre_t1.to(device), ce_t1.to(device)
                fake_ce = generator(pre_t1)
                
                dice = criterion_dice(fake_ce, ce_t1)
                val_dice_loss += dice.item()
                
                # Full validation loss (same composition as training for consistency)
                perceptual = criterion_perceptual(fake_ce, ce_t1)
                adv = criterion_adv(discriminator(fake_ce), torch.ones_like(discriminator(fake_ce)))
                val_loss += (dice + args.lambda_perceptual * perceptual + args.lambda_adv * adv).item()
        
        val_loss /= len(val_loader)
        val_dice_loss /= len(val_loader)
        
        print(f"[Validation] Loss: {val_loss:.4f} | Dice Loss: {val_dice_loss:.4f}")
        
        # Log validation metrics
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Dice_Loss', val_dice_loss, epoch)
        
        # Update learning rate schedulers
        scheduler_g.step(val_loss)
        scheduler_d.step(val_loss)
        
        # Check early stopping
        early_stopper(val_loss, generator)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'val_loss': val_loss,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'latest_checkpoint.pt'))
            print(f"Checkpoint saved (val_loss: {val_loss:.4f})")
        
        if early_stopper.early_stop:
            print("Early stopping triggered. Ending training.")
            break
    
    # Save final model
    torch.save(generator.state_dict(), os.path.join(args.save_dir, 'final_generator.pt'))
    print("Training completed. Final model saved.")
    writer.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        import traceback
        print(f"An error occurred during training: {e}")
        traceback.print_exc()