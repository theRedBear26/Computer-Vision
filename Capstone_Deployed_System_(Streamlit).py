import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🏗️ Concrete Crack Detector")
st.write("Upload an image of concrete to check if it contains a structural crack.")

# Load the model (Runs Project 11 first if not found)
try:
    model = tf.keras.models.load_model('crack_model.h5')
except:
    st.warning("Model not found. Please run `11_crack_detection.py` first to generate `crack_model.h5`.")
    model = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    st.write("---")
    if prediction > 0.5:
        st.error(f"⚠️ **Crack Detected!** (Confidence: {prediction*100:.2f}%)")
    else:
        st.success(f"✅ **No Crack Detected.** (Confidence: {(1-prediction)*100:.2f}%)")