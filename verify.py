import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from numpy import asarray, pad
from math import floor, ceil
import os
from model.metrics import euclidean_distance,constructive_loss
from data.preprocess import process_image
# ---------------- CONFIG ----------------
THRESHOLD = 0.98
MODEL_PATH ="model/BestModel.h5"


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    return load_model(
        MODEL_PATH,
        custom_objects={
            "euclidean_distance": euclidean_distance,
            "constructive_loss": constructive_loss
        }
    )

# ---------------- MAIN PAGE ----------------
def verify_page():

    st.title("Step 2: Verify Signature")

    # 🔒 Confidentiality Notice
    st.info(" Uploaded signatures are processed securely and not stored permanently.")

    model = load_my_model()

    test_file = st.file_uploader("Upload Signature to Verify", type=["png","jpg","jpeg"])

    if test_file:

        test_img = Image.open(test_file)
        st.image(test_img, caption="Test Signature", width=300)

        if st.button("Verify Signature"):

            with st.spinner("Analyzing..."):

                # Load reference
                ref_path = "uploads/reference/reference.png"

                if not os.path.exists(ref_path):
                    st.error("Reference signature not found!")
                    return

                ref_img = Image.open(ref_path)
                #ref = ref_img
                ref = process_image(ref_img)
                test = process_image(test_img)

                pred = model.predict([ref, test])[0][0]

            st.subheader(f"Difference Score: {pred:.4f}")

            # Decision
            print(pred)
            if pred <= THRESHOLD:
                st.error(" Forged Signature")
            else:
                st.success("Genuine Signature")

            # Confidence
            confidence = abs(pred - THRESHOLD)

            st.write(f"Confidence Level: {confidence:.2f}")

    # Back button
    if st.button("Back"):
        st.session_state.page = "upload"
        st.rerun()
