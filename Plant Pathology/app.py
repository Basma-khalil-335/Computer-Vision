# app.py
import streamlit as st # type: ignore
from PIL import Image, ImageOps # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from pathlib import Path
import streamlit as st
st.set_page_config(layout="wide")
# st.write("The app has started ✅")
# -----------------------------
# Configuration

st.set_page_config(page_title="Plant Pathology Detection", layout="centered", initial_sidebar_state="auto")

# Put your model file name here (make sure it's in the same folder or provide full path)
MODEL_PATH = "plant_pathology_model.keras"   # <-- change if your model filename is different
# Stable color (single color used across UI)
PRIMARY_COLOR = "#1f77b4"   # calm blue — change to any hex color you like
CARD_BG = "#f7fbff"

st.markdown(
    f"""
    <style>
    /* Page background */
    .stApp {{
        background-color: #ffffff;
    }}

    /* Header / Title bar */
    .app-header {{
        background: linear-gradient(90deg, {PRIMARY_COLOR}33, {PRIMARY_COLOR}11);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }}
    .app-title {{
        color: {PRIMARY_COLOR};
        font-size:28px;
        font-weight:600;
    }}
    /* Card style for main area */
    .card {{
        background: {CARD_BG};
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    /* Button color */
    .css-1emrehy.edgvbvh3 {{
        background-color: {PRIMARY_COLOR} !important;
        border-color: {PRIMARY_COLOR} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model(path: str):
    """Load and return compiled Keras model (cached)."""
    if not Path(path).exists():
        return None
    model = load_model(path)
    return model

def preprocess_image(image: Image.Image, target_size=(224,224)):
    """Preprocess PIL image for VGG19-based model used in training.
       Here we assume training used rescale 1./255 (not VGG preprocess_input).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return arr

def predict_image(model, image_array, class_names):
    """Return (predicted_label, probabilities)"""
    preds = model.predict(image_array)[0]   # shape (num_classes,)
    top_idx = int(np.argmax(preds))
    label = class_names[top_idx]
    return label, preds

# -----------------------------
# Main UI
# -----------------------------
model = load_trained_model(MODEL_PATH)

with st.container():
    st.markdown('<div class="app-header"><div class="app-title">🌿 Plant Pathology — VGG19 Classifier</div></div>', unsafe_allow_html=True)

col1,center , col2 = st.columns([1,2, 1])
with center:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader(
        "Choose an apple leaf image (jpg/png)",
        type=["jpg","jpeg","png"]
    )
    st.markdown("</div>", unsafe_allow_html=True)



if uploaded_file is not None:
    image = Image.open(uploaded_file)
    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        st.image(image, caption="Uploaded image", width=300)
        if model is None:
            st.warning("No model available to predict. Upload a trained model file to the app folder and restart.")
        else:
            with st.spinner("Preprocessing image..."):
                img_arr = preprocess_image(image, target_size=(224,224))
            with st.spinner("Running model prediction..."):
                class_names = list(model.class_names) if hasattr(model, "class_names") else ["healthy", "rust", "powdery"]
                label, probs = predict_image(model, img_arr, class_names)
            st.success(f"Predicted disease: **{label}**")



    

    if model is None:
        st.warning("No model available to predict. Upload a trained model file to the app folder and restart.")
    else:
        st.markdown("**Preprocessing & Prediction**")
        with st.spinner("Preprocessing image..."):
            img_arr = preprocess_image(image, target_size=(224,224))

        with st.spinner("Running model prediction..."):
            class_names = list(model.class_names) if hasattr(model, "class_names") else None

            # If model doesn't store class_names, we attempt a default mapping
            if class_names is None:
                # Change this list in the same order as your training generator class indices
                class_names = ["healthy", "rust", "powdery"]

            label, probs = predict_image(model, img_arr, class_names)

        # Show results
        st.success(f"Predicted: **{label}**")
        
else:
    st.info("Upload an image to see predictions. You can also drag & drop files.")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
