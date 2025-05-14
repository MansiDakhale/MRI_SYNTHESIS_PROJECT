import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.config import *
from utils.losses import dice_loss
from Unet3D_model import UNet3D  
from dataset import MRISegmentationDataset 

def save_model(model, path):
    torch.save(model.state_dict(), path)

def train():
    model = UNet3D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = dice_loss
    writer = SummaryWriter(LOG_DIR)
    scaler = torch.cuda.amp.GradScaler() if AMP else None

    dataset = MRISegmentationDataset(r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\Resized_dataset\train_resized")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=True, num_workers=NUM_WORKERS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')
    train_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")

        for ce, seg in progress_bar:
            ce, seg = ce.to(DEVICE), seg.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=AMP):
                pred = model(ce)
                loss = criterion(pred, seg)

            if AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{epoch_loss / (progress_bar.n + 1):.4f}"})

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        print(f"Epoch {epoch}: Avg Dice Loss = {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, os.path.join(CHECKPOINT_DIR, "best.pth"))

        # Save last model
        save_model(model, os.path.join(CHECKPOINT_DIR, "last.pth"))

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
    plt.title('Training Dice Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    writer.close()

if __name__ == "__main__":
    train()
