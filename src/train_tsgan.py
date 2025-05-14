

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import os

from Model.generator import TSGANGenerator
from Model.discriminator import TSGANDiscriminator
from utils.losses import dice_loss, perceptual_loss
from tsgan_dataset import TSGANDataset

# Multiprocessing fix
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

# Configs
def setup_training():
    epochs = 50
    lr = 5e-5
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs/tsgan_training/images', exist_ok=True)

    return epochs, lr, batch_size, device

def main():
    epochs, lr, batch_size, device = setup_training()

    train_dataset = TSGANDataset(
        r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\Resized_dataset\train_resized"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    G = TSGANGenerator(in_channels=2, out_channels=1).to(device)
    D = TSGANDiscriminator(in_channels=3).to(device)

    opt_G = optim.AdamW(G.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    opt_D = optim.AdamW(D.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(opt_G, mode='min', factor=0.5, patience=5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(opt_D, mode='min', factor=0.5, patience=5)

    bce = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir="runs/tsgan_training")

    best_loss = float("inf")
    patience = 10
    trigger = 0

    for epoch in range(epochs):
        G.train()
        D.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_G_loss = 0.0
        total_D_loss = 0.0

        for pret1, cet1, seg in loop:
            # Ensure correct shape: [B, 1, D, H, W]
            if pret1.dim() == 6:
                pret1 = pret1.squeeze(2)
            if cet1.dim() == 6:
                cet1 = cet1.squeeze(2)
            if seg.dim() == 6:
                seg = seg.squeeze(2)        
            pret1, cet1, seg = pret1.to(device), cet1.to(device), seg.to(device)
            
            
            print("pret1 shape:", pret1.shape)
            print("seg shape before squeeze:", seg.shape)

            if seg.dim() == 6:
                seg = seg.squeeze(2)  # Squeeze [B, 1, 1, D, H, W] -> [B, 1, D, H, W]

            print("seg shape after squeeze:", seg.shape)

            # Check shape match
            assert pret1.shape == seg.shape, f"Shape mismatch: pret1 {pret1.shape}, seg {seg.shape}"
            input_G = torch.cat([pret1, seg], dim=1)
            input_D_real = torch.cat([pret1, seg, cet1], dim=1)

            # -----------------------
            # Discriminator Training
            # -----------------------
            opt_D.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                fake_cet1 = G(input_G)
                D_real = D(input_D_real)
                D_fake = D(torch.cat([pret1, seg, fake_cet1.detach()], dim=1))

                loss_D_real = bce(D_real, torch.ones_like(D_real))
                loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))
                loss_D = (loss_D_real + loss_D_fake) / 2

            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            # -----------------------
            # Generator Training
            # -----------------------
            opt_G.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                fake_cet1 = G(input_G)
                D_fake_G = D(torch.cat([pret1, seg, fake_cet1], dim=1))

                adv_loss = bce(D_fake_G, torch.ones_like(D_fake_G))
                d_loss = dice_loss(fake_cet1, cet1)
                p_loss = perceptual_loss(fake_cet1, cet1)

                loss_G = d_loss + 0.01 * adv_loss + 0.1 * p_loss

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()
            loop.set_postfix({
                "G_loss": f"{total_G_loss/(loop.n+1):.4f}",
                "D_loss": f"{total_D_loss/(loop.n+1):.4f}"
            })

        avg_G_loss = total_G_loss / len(train_loader)
        avg_D_loss = total_D_loss / len(train_loader)

        writer.add_scalar("Loss/Generator", avg_G_loss, epoch)
        writer.add_scalar("Loss/Discriminator", avg_D_loss, epoch)

        # Adjust learning rate
        scheduler_G.step(avg_G_loss)
        scheduler_D.step(avg_D_loss)

        # Save best model
        if avg_G_loss < best_loss:
            best_loss = avg_G_loss
            trigger = 0
            torch.save(G.state_dict(), "checkpoints/best_generator.pth")
            torch.save(D.state_dict(), "checkpoints/best_discriminator.pth")
        #else:
         #   trigger += 1
          #  if trigger >= patience:
           #     print(f"Early stopping triggered at epoch {epoch+1}")
            #    break

        # Save a 2D slice from the center slice of volume
        with torch.no_grad():
            G.eval()
            sample_output = G(input_G[:2])
            mid_slice = sample_output.shape[2] // 2
            slice_grid = make_grid(sample_output[:, :, mid_slice], normalize=True)
            writer.add_image("Synthesized_CeT1", slice_grid, global_step=epoch)

    writer.close()

if __name__ == '__main__':
    main()










