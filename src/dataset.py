import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class MRISegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.root_dir = root_dir
        self._load_dataset()

    def _load_dataset(self):
        subjects = os.listdir(self.root_dir)
        for subject in subjects:
            subject_path = os.path.join(self.root_dir, subject)
            ce_path = os.path.join(subject_path, 'CeT1')
            seg_path = os.path.join(subject_path, 'Segmentation')

            # Skip if folders don't exist
            if not os.path.isdir(ce_path) or not os.path.isdir(seg_path):
                print(f"[WARNING] Missing CeT1 or Segmentation folder for subject: {subject}")
                continue

            ce_files = [f for f in os.listdir(ce_path) if f.endswith('.nii.gz')]
            seg_files = [f for f in os.listdir(seg_path) if f.endswith('.nii.gz')]

            for ce_file, seg_file in zip(sorted(ce_files), sorted(seg_files)):
                ce_full_path = os.path.join(ce_path, ce_file)
                seg_full_path = os.path.join(seg_path, seg_file)

                if os.path.exists(ce_full_path) and os.path.exists(seg_full_path):
                    self.samples.append((ce_full_path, seg_full_path))
                else:
                    print(f"[WARNING] Missing file pair: {ce_full_path}, {seg_full_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ce_path, seg_path = self.samples[idx]
        ce_img = nib.load(ce_path).get_fdata()
        seg_img = nib.load(seg_path).get_fdata()

        ce_tensor = torch.tensor(ce_img, dtype=torch.float32).unsqueeze(0)
        seg_tensor = torch.tensor(seg_img, dtype=torch.float32).unsqueeze(0)

        return ce_tensor, seg_tensor