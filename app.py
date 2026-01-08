import os

# Force CPU usage by hiding GPU devices and disabling XLA flags that might cause errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
from skimage.color import label2rgb
import matplotlib.pyplot as plt

# Import model architecture and analysis logic from nuclei_segmentation.py
from nuclei_segmentation import build_unet, analyze_nuclei_size

# --- CONFIGURATION ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
WEIGHTS_PATH = 'weights/model_dsbowl_epoch_50.keras'
PIXEL_CALIBRATION = 0.5  # 1 pixel = 0.5 microns (same as in nuclei_segmentation.py)

st.set_page_config(page_title="Nuclei Segmentation CLI", layout="wide")

st.title("🔬 Breast Scan Nuclei Segmentation")
st.markdown("""
Upload a breast scan image to identify and analyze nuclei using a pre-trained U-Net model.
""")

@st.cache_resource
def load_model():
    # Ensure model is built and weights are loaded on CPU
    with tf.device('/CPU:0'):
        model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        if os.path.exists(WEIGHTS_PATH):
            model.load_weights(WEIGHTS_PATH)
        else:
            st.error(f"Weights file not found at {WEIGHTS_PATH}")
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose a breast scan image...", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    # 1. Load and display original image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    # Normalize if image is not 8-bit (e.g., 16-bit TIFF)
    if img_array.max() > 255:
        img_array = (img_array / 65535.0 * 255).astype(np.uint8)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, width='stretch')
    
    # 2. Preprocessing
    img_resized = resize(img_array, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img_input = np.expand_dims(img_resized, axis=0)

    # Sidebar Options
    st.sidebar.header("Parameters")
    threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # 3. Inference
    with st.spinner('Analyzing scan...'):
        with tf.device('/CPU:0'):
            preds = model.predict(img_input, verbose=0)
        pred_mask = preds[0, :, :, 0]
        
        # 4. Post-processing and Analysis
        labeled_nuclei, stats = analyze_nuclei_size(
            img_resized, 
            pred_mask, 
            pixel_to_micron_ratio=PIXEL_CALIBRATION,
            threshold=threshold
        )
        
        # Create overlay
        image_label_overlay = label2rgb(labeled_nuclei, image=img_resized/255.0, bg_label=0)
    
    with col2:
        st.subheader("Segmentation Overlay")
        st.image(image_label_overlay, width='stretch', clamp=True)
        
    # Visualize Raw Probability Mask
    with st.expander("View Raw Prediction Mask"):
        st.write("This map shows the model's confidence for each pixel (Brighter = Higher Probability).")
        st.image(pred_mask, clamp=True, width='stretch')
        
    # 5. Results & Statistics
    st.divider()
    st.subheader(f"📊 Analysis Results: {len(stats)} Nuclei Detected")
    
    if stats:
        # Display summary metrics
        avg_area = np.mean([s['area_sq_microns'] for s in stats])
        avg_diameter = np.mean([s['diameter_microns'] for s in stats])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Nuclei Count", len(stats))
        m2.metric("Avg Area", f"{avg_area:.2f} µm²")
        m3.metric("Avg Diameter", f"{avg_diameter:.2f} µm")
        
        # Display detailed table
        st.write("Detailed Nuclei Data:")
        st.dataframe(stats)
    else:
        st.warning("No nuclei detected in this scan.")

else:
    st.info("Please upload an image to start the analysis.")

st.sidebar.header("About")
st.sidebar.info("""
This tool uses a U-Net deep learning model trained on the Data Science Bowl 2018 dataset.
It identifies nuclei in histological images and calculates morphological statistics.
""")
