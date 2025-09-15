from tkinter import Image
import streamlit as st
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array   # type: ignore
import numpy as np
from PIL import Image


# --- Load model once ---
@st.cache_resource
def load_mask_model():
    return load_model("vgg19_masks.keras")

model = load_mask_model()

st.title("😷 Mask Detection App")
st.write("Upload an image to check if the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    from PIL import Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess to (224,224) to match VGG19
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    # If model outputs a single sigmoid neuron (0=no mask, 1=mask)
    label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
