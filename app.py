import streamlit as st
from verify import verify_page 
from data.preprocess import process_image

import os
from PIL import Image, ImageOps
import numpy as np
from numpy import asarray, pad
from math import floor, ceil

# SESSION STATE
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ---------------- PAGE 1 ----------------
if st.session_state.page == "upload":




    st.title("Signature Verification System")
    st.header("Step 1: Upload Reference Signature")

    uploaded_file = st.file_uploader("Upload Real Signature", type=["png","jpg","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        #image = np.array(image)

        #processed_img = preprocess_to_dataset_style(image)

        if st.button("Save Signature"):
            os.makedirs("uploads/reference/", exist_ok=True)
            #Image.fromarray(processed_img.squeeze()).save("uploads/reference/reference.png")
            image.save("uploads/reference/reference.png")

            st.success("Signature saved successfully!")

            st.session_state.page = "verify"
            st.rerun()

# ---------------- PAGE 2 ----------------
elif st.session_state.page == "verify":
    verify_page()
