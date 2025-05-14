import os
import shutil
import pandas as pd

metadata_csv = r"c:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\TumorDataset\manifest-PyHQgfru6393647793776378748\metadata.csv"

# Function to organize DICOM files based on Series Description
def organize_dicom_files(metadata_csv, output_folder="data/raw"):
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f" ERROR: '{metadata_csv}' not found! Check the file path.")

    # Load metadata CSV
    df = pd.read_csv(metadata_csv)
    
    # Debugging: Print actual column names
    print(" Columns in metadata.csv:", df.columns)

    required_columns = ["Subject ID", "Series Description", "File Location"]

    # Check if required columns exist
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f" ERROR: Missing column '{col}' in metadata.csv! Available columns: {df.columns}")

    # Set base directory (where metadata.csv is located)
    base_dir = os.path.dirname(metadata_csv)

    for _, row in df.iterrows():
        patient_id = str(row["Subject ID"])
        series_desc = str(row["Series Description"]).lower()  # Convert to lowercase

        # Resolve absolute file path relative to metadata.csv
        source_path = os.path.abspath(os.path.join(base_dir, str(row.get("File Location", "")).strip()))

        # Debugging: Print file path being checked
        print(f" Checking: {source_path}")

        # Skip files that don't exist
        if not os.path.exists(source_path):
            print(f" File Not Found: {source_path}")
            continue

        # Keywords for Pre-Contrast (PreT1) images
        pre_contrast_keywords = [
            "pret1", "pre-contrast", "precontrast", "t1 pre", "t1-pre",
            "t1 precontrast", "non-contrast", "t1w pre", "t1-weighted pre",
            "pre t1", "before contrast", "T1", "T2", "scout", "locator", "axial-locator", "SCOUT"
        ]

        # Keywords for Post-Contrast (CeT1) images
        post_contrast_keywords = [
            "cet1", "post-contrast", "postcontrast", "t1 post", "t1-post",
            "t1 postcontrast", "contrast enhanced", "gadolinium",
            "t1w post", "t1-weighted post", "after contrast", "t1ce", "dynamic-3dfgre ser", "dynamic-3dfgre pe1", "VOI PE", "dynamic", "CTLMIDSag3DSPGRVBw GRx SER", "CTLMIDSag3DSPGRVBw GRx PE1", "PELVICSag3DSPGRVBw GRx PE1", "pe1"
        ]

        # Keywords for Segmentation/Mask images
        segmentation_keywords = [
            "segmentation", "mask", "tumor mask", "roi", "annotation",
            "ground truth", "label", "lesion mask", "binary mask",
            "tumor segmentation", "mri segmentation"
        ]

        # Determine category based on keywords
        if any(keyword in series_desc for keyword in pre_contrast_keywords):
            category = "PreT1"
        elif any(keyword in series_desc for keyword in post_contrast_keywords):
            category = "CeT1"
        elif any(keyword in series_desc for keyword in segmentation_keywords):
            category = "Segmentation"
        else:
            print(f" Skipping Unrecognized Series: {series_desc}")
            continue  # Skip files that don't match any category

        # Define target directory
        target_dir = os.path.join(output_folder, patient_id, category)
        os.makedirs(target_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Move file to categorized folder
        destination_path = os.path.join(target_dir, os.path.basename(source_path))
        shutil.move(source_path, destination_path)
        print(f" Moved: {source_path} ➝ {destination_path}")

    print(" DICOM files organized successfully!")

# Run file organization
output_folder = r"C:\Users\OMEN\Desktop\MRI_Synthesis_Project\MRI_SYNTHESIS_PROJECT\data\raw"
organize_dicom_files(metadata_csv, output_folder)
