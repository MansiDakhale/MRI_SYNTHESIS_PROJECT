import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from tsgan import UNet3DGenerator
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_DIR = r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\dataset\val"

# Load validation set
val_dataset = MRIDataset(base_dir=VAL_DIR)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load trained model
generator = UNet3DGenerator().to(DEVICE)
generator.load_state_dict(torch.load("outputs/checkpointss/generator_epoch100.pth"))
generator.eval()

# Validation loop
with torch.no_grad():
    for idx, (pret1, cet1, seg) in enumerate(val_loader):
        pret1, cet1, seg = pret1.to(DEVICE), cet1.to(DEVICE), seg.to(DEVICE)
        fake_cet1 = generator(pret1, seg)

        # Save visual slice or compute metrics
        # e.g., save middle slice or compare real vs fake

print("Validation done.")
