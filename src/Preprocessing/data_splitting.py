import os
import shutil
import random

def split_dataset(input_folder, output_folder, train_ratio=0.7, val_ratio=0.2):
    os.makedirs(output_folder, exist_ok=True)

    for patient in os.listdir(input_folder):  # Loop through patient IDs
        patient_folder = os.path.join(input_folder, patient)

        for category in ["PreT1", "CeT1", "Segmentation"]:
            category_path = os.path.join(patient_folder, category)

            if not os.path.exists(category_path):  # Skip if folder doesn't exist
                print(f" Skipping missing category: {category_path}")
                continue

            files = os.listdir(category_path)
            random.shuffle(files)

            train_split = int(len(files) * train_ratio)
            val_split = int(len(files) * (train_ratio + val_ratio))

            for split, name in zip(
                [files[:train_split], files[train_split:val_split], files[val_split:]],
                ["train", "val", "test"]
            ):
                split_path = os.path.join(output_folder, name, patient, category)
                os.makedirs(split_path, exist_ok=True)

                for file in split:
                    shutil.move(os.path.join(category_path, file), os.path.join(split_path, file))

                print(f"{name.capitalize()} Set: {len(split)} {category} images")

# Run dataset splitting
split_dataset(r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\preprocessed", r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\dataset")
