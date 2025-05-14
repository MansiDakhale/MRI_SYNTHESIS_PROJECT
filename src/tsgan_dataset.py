import os
import glob
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)

class TSGANDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        logging.info(f"🔍 Scanning TSGAN dataset in: {root_dir}")

        patient_dirs = sorted(os.listdir(root_dir))

        for patient_id in patient_dirs:
            patient_path = os.path.join(root_dir, patient_id)

            pre_t1_path = os.path.join(patient_path, 'PreT1')
            cet1_path = os.path.join(patient_path, 'CeT1')
            seg_path = os.path.join(patient_path, 'Segmentation')

            # Glob the .nii.gz files
            pre_t1_file = glob.glob(os.path.join(pre_t1_path, '*.nii.gz'))
            cet1_file = glob.glob(os.path.join(cet1_path, '*.nii.gz'))
            seg_file = glob.glob(os.path.join(seg_path, '*.nii.gz'))

            if not (pre_t1_file and cet1_file and seg_file):
                logging.warning(f"[SKIPPED] Missing one or more files in {patient_id}")
                continue

            self.samples.append({
                'pre_t1': pre_t1_file[0],
                'cet1': cet1_file[0],
                'seg': seg_file[0],
                'patient_id': patient_id,
            })

        if len(self.samples) == 0:
            raise ValueError("❌ No valid samples found. Please check dataset structure!")

        logging.info(f"✅ Loaded {len(self.samples)} valid patient volumes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pre_t1 = nib.load(sample['pre_t1']).get_fdata().astype(np.float32)
        cet1 = nib.load(sample['cet1']).get_fdata().astype(np.float32)
        seg = nib.load(sample['seg']).get_fdata().astype(np.float32)

        # Add channel dimension -> [1, D, H, W]
        pre_t1 = np.expand_dims(pre_t1, axis=0)
        cet1 = np.expand_dims(cet1, axis=0)
        seg = np.expand_dims(seg, axis=0)

        if self.transform:
            pre_t1 = self.transform(pre_t1)
            cet1 = self.transform(cet1)
            seg = self.transform(seg)

        # Convert to PyTorch tensors
        pre_t1_tensor = torch.tensor(pre_t1, dtype=torch.float32).unsqueeze(0)
        cet1_tensor = torch.tensor(cet1, dtype=torch.float32).unsqueeze(0)
        seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
        return pre_t1_tensor, cet1_tensor, seg_tensor
