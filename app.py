# DEBUG
import os
print("Current Working Directory:", os.getcwd())
print("Models folder exists:", os.path.exists("models"))
print("Model file exists:", os.path.exists("models/custom_cnn.keras"))

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cache models so they are loaded only once
@st.cache_resource
def load_models():
  models = {
    "Custom CNN": load_model("/workspaces/malaria-image-classification/models/custom_cnn.keras"),
    "MobileNetV2 (Frozen)": load_model("/workspaces/malaria-image-classification/models/malaria_mobilenetv2_finetuned.keras"),
    "MobileNetV2 (Fine-tuned)": load_model("/workspaces/malaria-image-classification/models/malaria_mobilenetv2_frozen.keras")
  }
  return models

# Load models
models = load_models()

# Streamlit UI
st.title("Malaria Cell Detection App")
st.write("Upload a blood cell image and select a model to classify it.")

# Model selection dropdown
model_choice = st.selectbox(
  "Choose a model:",
  list(models.keys())
)

model = models[model_choice]

# File uploader
file = st.file_uploader("Upload a Cell Image", type=['jpg', 'png', 'jpeg'])

if file is not None:
  # Preprocess image
  img = image.load_img(file, target_size=(128, 128))
  img_array = np.expand_dims(image.img_to_array(img)/255.0, axis=0)

  # Prediction
  pred = model.predict(img_array)[0][0]
  confidence = pred if pred > 0.5 else 1 - pred

  st.image(img, caption="Uploaded Cell Image", use_container_width=True)
  
  # Display results
  if pred > 0.5:
    st.success(f"✅ Uninfected Cell Detected Confidence {confidence:.2%}")
  else:
    st.error(f"❌ Infected Cell Detected Confidence {confidence:.2%}")

  st.write(f"Raw model output: {pred:.4f}")