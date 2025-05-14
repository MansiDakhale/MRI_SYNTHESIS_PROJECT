import os
import dicom2nifti

def dicom_to_nifti(input_folder, output_folder):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f" ERROR: The folder '{input_folder}' does not exist!")

    os.makedirs(output_folder, exist_ok=True)

    for patient in os.listdir(input_folder):  # Loop through patient folders
        patient_folder = os.path.join(input_folder, patient)

        for category in ["PreT1", "CeT1", "Segmentation"]:
            input_path = os.path.join(patient_folder, category)
            output_path = os.path.join(output_folder, patient, category)

            if not os.path.exists(input_path):  # Check if category folder exists
                print(f"WARNING: '{input_path}' folder is missing. Skipping...")
                continue  

            os.makedirs(output_path, exist_ok=True)

            # Convert DICOM to NIfTI
            dicom2nifti.convert_directory(input_path, output_path)

            print(f"Converted: {input_path} → {output_path}")

# Run conversion
dicom_to_nifti(r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\raw", 
               r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\nifti")

