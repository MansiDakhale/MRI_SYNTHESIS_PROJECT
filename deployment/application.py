import streamlit as st
import numpy as np
import nibabel as nib
import os
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the generate_ceT1 function from the src module
from src.generate_mri import generate_ceT1

st.set_page_config(layout="centered")
st.title("TSGAN: Contrast-Free MRI Synthesizer")

st.write("Upload PreT1 and Segmentation volumes to synthesize CeT1 MRI.")

pret1_file = st.file_uploader("Upload PreT1 (.nii.gz)", type=["nii", "gz"])
seg_file = st.file_uploader("Upload Segmentation (.nii.gz)", type=["nii", "gz"])
real_cet1_file = st.file_uploader("Optional: Upload Real CeT1 for Comparison (.nii.gz)", type=["nii", "gz"])

if st.button("Generate CeT1") and pret1_file and seg_file:
    with open("temp_pret1.nii.gz", "wb") as f:
        f.write(pret1_file.read())
    with open("temp_seg.nii.gz", "wb") as f:
        f.write(seg_file.read())

    st.info("Synthesizing CeT1... 🔄")
    cet1_output = generate_ceT1("temp_pret1.nii.gz", "temp_seg.nii.gz")  # numpy array [D, H, W]

    # Let user select slice
    slice_idx = st.slider("Select Slice Index", 0, cet1_output.shape[0] - 1, cet1_output.shape[0] // 2)

    col1, col2 = st.columns(2)

    with col1:
        st.image(cet1_output[slice_idx], caption=f"Synthesized CeT1 (Slice {slice_idx})", clamp=True)

    if real_cet1_file:
        with open("temp_real_cet1.nii.gz", "wb") as f:
            f.write(real_cet1_file.read())
        real = nib.load("temp_real_cet1.nii.gz").get_fdata()
        with col2:
            st.image(real[slice_idx], caption=f"Real CeT1 (Slice {slice_idx})", clamp=True)

    # Save output
    output_nifti = nib.Nifti1Image(cet1_output, affine=np.eye(4))
    nib.save(output_nifti, "synthesized_cet1.nii.gz")

    with open("synthesized_cet1.nii.gz", "rb") as f:
        st.download_button(
            label="📥 Download Synthesized CeT1 (.nii.gz)",
            data=f,
            file_name="synthesized_cet1.nii.gz",
            mime="application/gzip"
        )

    st.success("Synthesis Complete! ✅")
