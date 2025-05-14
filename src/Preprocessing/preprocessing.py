import nibabel as nib
import numpy as np
import os
from skimage.transform import resize

def preprocess_nifti(input_folder, output_folder, target_shape=(256, 256, 64)):
    os.makedirs(output_folder, exist_ok=True)

    for patient in os.listdir(input_folder):  # Loop through patient folders
        patient_folder = os.path.join(input_folder, patient)
        
        for category in ["PreT1", "CeT1", "Segmentation"]:
            input_path = os.path.join(patient_folder, category)
            output_path = os.path.join(output_folder, patient, category)  # Save per patient

            if not os.path.exists(input_path):  # Check before processing
                print(f" WARNING: '{input_path}' does not exist. Skipping...")
                continue  

            os.makedirs(output_path, exist_ok=True)

            for file in os.listdir(input_path):
                nifti_path = os.path.join(input_path, file)
                
                try:
                    img = nib.load(nifti_path).get_fdata()

                    # Ensure valid data
                    if img is None or np.max(img) == np.min(img):
                        print(f" Skipping {file} due to empty or invalid data")
                        continue

                    epsilon = 1e-8  # Prevent division by zero

                    if category == "Segmentation":
                    # Resize with nearest-neighbor interpolation (order=0)
                        img_resized = resize(img, target_shape, order=0, preserve_range=True, anti_aliasing=False)
                    else:
                    # Normalize intensity to [-1, 1] and resize with interpolation
                        img = (img - np.min(img)) / (np.max(img) - np.min(img) + epsilon)
                        img = (img * 2) - 1
                        img_resized = resize(img, target_shape, anti_aliasing=True, preserve_range=True)


                    # Save processed image
                    nifti_output = os.path.join(output_path, file)
                    nib.save(nib.Nifti1Image(img_resized, np.eye(4)), nifti_output)

                    print(f" Processed: {file} → {nifti_output}")

                except Exception as e:
                    print(f" ERROR processing {file}: {str(e)}")
                    continue  # Skip the file if it fails

# Run preprocessing
preprocess_nifti(r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\nifti", r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\preprocessed")


