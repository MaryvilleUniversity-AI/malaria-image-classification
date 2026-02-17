import os

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

print(cv2.__version__)

# Grad-CAM heatmap generator
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
  try:
    grad_model = tf.keras.models.Model(
      inputs=model.input,
      outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
  
    with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(img_array)
      loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

  except Exception:
    # Fallback for simple Sequential
    input_tensor = tf.keras.Input(shape=img_array.shape[1:])
    x = input_tensor
    conv_outputs = None

    for layer in model.layers:
      x = layer(x)
      if layer.name == last_conv_layer_name:
        conv_output = x

    if conv_output is None:
      raise ValueError(f"Layer {last_conv_layer_name} not found.")

    replay_model = tf.keras.Model(inputs=input_tensor, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
      conv_output, predictions = replay_model(img_array)
      loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# Overlay display helper (returns image)
def overlay_gradcam_full(img, heatmap, alpha=0.4):
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# Preprocess
def preprocess_for_model(img, model_choice):
  if "MobileNetV2" in model_choice:
    # Use MobileNetV2 preprocessing & resize to expected 128x128
    img = img.resize((128,128))
    arr = np.array(img)
    arr = mobilenet_preprocess(arr)
    return np.expand_dims(arr, axis=0)
  else:
    # Custom CNN uses simple 0-1 scale
    img = img.resize((128,128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

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
  # Display original upload
  image = Image.open(file).convert("RGB")
  st.image(image, caption="Uploaded Cell Image", use_column_width=True)

  # Preprocess image for selected model
  img_array = preprocess_for_model(image, model_choice)

  # --- Make Prediction ---
  pred = model.predict(img_array)[0][0]
  confidence = pred if pred > 0.5 else 1 - pred
  pred_class = "Uninfected" if pred > 0.5 else "Infected"

  st.write(f"**Prediction:** {pred_class} ({confidence:.2%} confidence)")
  st.write(f"Raw model output: {pred:.4f}")

  # --- Find Last Convolutional Layer ---
  if model_choice == "Custom CNN":
    last_conv_layer_name = "conv2d_1"
  else:
    last_conv_layer_name = "Conv_1"
  
  try:
    # Compute Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
  except Exception as e:
    st.error(f"Grad-CAM failed: {e}")
  else:
    # --- Display 128x128 Overlay ---
    small_img = np.array(image.resize((128,128)))
    heatmap_small = cv2.resize(heatmap, (128,128))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_small), cv2.COLORMAP_JET)
    overlay_small = cv2.addWeighted(small_img, 0.6, heatmap_color, 0.4, 0)

    st.image(overlay_small, caption="Grad-CAM Overlay (128x128)", use_column_width=True)

    # --- Display Full Resolution Overlay ---
    full_img = np.array(image)
    overlay_full = overlay_gradcam_full(full_img, heatmap)
    st.image(overlay_full, caption="Grad-CAM Overlay (Original size)", use_column_width=True)

  # Preprocess image
  # img = image.load_img(file, target_size=(128, 128))
  # img_array = np.expand_dims(image.img_to_array(img)/255.0, axis=0)

  # # Prediction
  # pred = model.predict(img_array)[0][0]
  # confidence = pred if pred > 0.5 else 1 - pred

  # st.image(img, caption="Uploaded Cell Image", use_container_width=True)
  
  # # Display results
  # if pred > 0.5:
  #   st.success(f"✅ Uninfected Cell Detected Confidence {confidence:.2%}")
  # else:
  #   st.error(f"❌ Infected Cell Detected Confidence {confidence:.2%}")

  # st.write(f"Raw model output: {pred:.4f}")