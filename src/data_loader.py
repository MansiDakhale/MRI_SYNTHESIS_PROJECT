from glob import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

import sys
print(sys.executable)



class MRIDataset(Dataset):
    def __init__(self, root_dir="data", split="train", transform=None):
        self.pre_paths = sorted(glob(os.path.join(root_dir, split, "pre_t1", "*.nii.gz")))
        self.ce_paths = sorted(glob(os.path.join(root_dir, split, "ce_t1", "*.nii.gz")))
        self.seg_paths = sorted(glob(os.path.join(root_dir, split, "seg", "*.nii.gz")))
        self.transform = transform

        assert len(self.pre_paths) == len(self.ce_paths) == len(self.seg_paths), "Mismatch in dataset size"

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre = nib.load(self.pre_paths[idx]).get_fdata().astype(np.float32)
        ce = nib.load(self.ce_paths[idx]).get_fdata().astype(np.float32)
        seg = nib.load(self.seg_paths[idx]).get_fdata().astype(np.float32)

        pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
        ce = (ce - ce.min()) / (ce.max() - ce.min() + 1e-8)

        pre = np.expand_dims(pre, axis=0)
        ce = np.expand_dims(ce, axis=0)
        seg = np.expand_dims(seg, axis=0)

        subject = tio.Subject(
            pre=tio.ScalarImage(tensor=pre),
            ce=tio.ScalarImage(tensor=ce),
            seg=tio.LabelMap(tensor=seg)
        )

        if self.transform:
            subject = self.transform(subject)

        pre = subject['pre'].data
        ce = subject['ce'].data
        seg = subject['seg'].data

        return pre, ce, seg

