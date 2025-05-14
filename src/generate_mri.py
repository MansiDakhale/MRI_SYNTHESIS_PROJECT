import torch
from tsgan.generator import TSGANGenerator
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = TSGANGenerator(in_channels=2, out_channels=1).to(device)
model.load_state_dict(torch.load("checkpoints/best_generator.pth", map_location=device))
model.eval()

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def preprocess(pret1_path, seg_path):
    pret1 = load_nifti(pret1_path)
    seg = load_nifti(seg_path)
    pret1 = torch.tensor(pret1, dtype=torch.float32).unsqueeze(0)
    seg = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
    input_tensor = torch.cat([pret1, seg], dim=0).unsqueeze(0).to(device)  # [1, 2, D, H, W]
    return input_tensor

def generate_ceT1(pret1_path, seg_path):
    input_tensor = preprocess(pret1_path, seg_path)
    with torch.no_grad():
        output = model(input_tensor)
    output = torch.sigmoid(output.squeeze()).cpu().numpy()  # [D, H, W]
    return output

def show_middle_slice(volume):
    mid = volume.shape[0] // 2
    plt.imshow(volume[mid], cmap='gray')
    plt.axis('off')
    plt.title("Synthesized CeT1")
    plt.show()
