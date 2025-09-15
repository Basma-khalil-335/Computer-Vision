# 🩺 FaceMask Detection

A deep learning project that classifies whether a person in an **uploaded image** is **wearing a face mask** or **not wearing a mask**.  
The model is built using **VGG19** with transfer learning for high accuracy.

**Model Performance:**
- Training Accuracy: 96.19%
- Validation Accuracy: 97.47%

---

## 🚀 Features
- Upload an image and instantly receive a **Mask** / **No Mask** prediction  
- Transfer learning with **VGG19** for strong performance  
- Simple Streamlit web interface for easy use

---

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras (VGG19)**
- **Streamlit** for the web interface
- Supporting libraries: Numpy, Pandas, Matplotlib

---

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Basma-khalil-335/Computer-Vision.git
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv env
   source env/bin/activate   # (Windows: env\Scripts\activate)
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Use the app**  
   - Open the provided local URL in your browser.  
   - Upload an image with a face.  
   - View the prediction result: **Mask** or **No Mask**.

---

## 📂 Project Structure
```
Facemask-Detection/
│
├─ app.py                # Streamlit web app
├─ model/                # Trained VGG19 model weights
├─ requirements.txt
└─ README.md
```

---

## 🧑‍💻 How It Works
1. The user uploads an image containing a face.  
2. The image is preprocessed (resized, normalized).  
3. A fine-tuned **VGG19** model predicts whether the person is **wearing a mask** or **not**.

---

## 🖼️ Screenshots
<img width="521" height="415" alt="mask detection" src="https://github.com/user-attachments/assets/4b4c4cc2-9343-4056-b18d-248196b02b3e" />
<img width="541" height="414" alt="notmask detection" src="https://github.com/user-attachments/assets/544cec56-b847-43b5-93cd-ed5bdfa5603f" />

---
## 🙌 Acknowledgements
- Dataset: [https://www.kaggle.com/datasets/omkargurav/face-mask-dataset]

