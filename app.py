import os

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Clear all cached resources
st.cache_resource.clear()

# Custom CNN Architecture
def build_custom_cnn(input_shape=(128, 128,3)):
  inputs = tf.keras.Input(shape=input_shape)

  x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)

  x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  return model

# MobileNetV2 (Frozen)
def build_mobilenetv2_frozen():
  base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
  )

  base_model.trainable = False

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.5)(x)
  output = Dense(1, activation="sigmoid")(x)
  model = Model(inputs=base_model.input, outputs=output)
  return model

# MobileNetV2 (Fine-Tuned)
def build_mobilenetv2_finetuned(input_shape=(128, 128,3)):
  base_model = MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet"
  )

  # Unfreeze last 30 layers
  for layer in base_model.layers[:-30]:
    layer.trainable = False
  for layer in base_model.layers[-30:]:
    layer.trainable = True

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.5)(x)
  output = Dense(1, activation="sigmoid")(x)
  
  model = Model(inputs=base_model.input, outputs=output)

  return model


# Grad-CAM heatmap generator
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
  # Create model that outputs both the conv layer and predictions
  grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
      model.get_layer(last_conv_layer_name).output,
      model.output
    ],
  )

  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array, training=False)
    loss = predictions[:, 0]
  
  grads = tape.gradient(loss, conv_outputs)

  # Global average pooling of gradients
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  # Normalize
  heatmap = tf.maximum(heatmap, 0)
  heatmap /= (tf.reduce_max(heatmap) + 1e-8)

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
  # --- Custom CNN ---
  custom_model = build_custom_cnn()
  custom_model.load_weights("/workspaces/malaria-image-classification/models/custom_cnn_weights_only.weights.h5")

  # --- MobileNetV2 Frozen ---
  mobilenet_frozen = build_mobilenetv2_frozen()
  mobilenet_frozen.load_weights("/workspaces/malaria-image-classification/models/malaria_mobilenetv2_frozen_weights_only.weights.h5")

  # --- MobileNetV2 Fine-Tuned ---
  mobilenet_finetuned = build_mobilenetv2_finetuned()
  mobilenet_finetuned.load_weights("/workspaces/malaria-image-classification/models/malaria_mobilenetv2_finetuned_weights_only.weights.h5")

  return {
    "Custom CNN": custom_model,
    "MobileNetV2 (Frozen)": mobilenet_frozen,
    "MobileNetV2 (Fine-tuned)": mobilenet_finetuned
  }

# Load models
models_dict = load_models()

# Streamlit UI
st.title("Malaria Cell Detection App")
st.write("Upload a blood cell image and select a model to classify it.")

# Model selection dropdown
selected_model_name = st.selectbox(
  "Choose a model:",
  list(models_dict.keys())
)

model = models_dict[selected_model_name]

# File uploader
file = st.file_uploader("Upload a Cell Image", type=['jpg', 'png', 'jpeg'])

if file:
  image = Image.open(file).convert("RGB")
  st.image(image, caption="Uploaded Image", width=400)

  # Preprocess
  img_array = np.array(image.resize((128, 128)))
  if "MobileNetV2" in selected_model_name:
    img_array = mobilenet_preprocess(img_array)
  else:
    img_array = img_array / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  # Predict
  pred = model.predict(img_array)[0][0]
  pred_class = "Uninfected" if pred > 0.5 else "Infected"
  confidence = pred if pred > 0.5 else 1 - pred

  st.write(f"**Prediction:** {pred_class} ({confidence:.2%} confidence)")

  # Select correct last conv layer
  if "Custom CNN" in selected_model_name:
    last_conv_layer_name = "conv2d_1"
  else:
    last_conv_layer_name = "Conv_1"

  try:
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
    heatmap_colored = cv2.applyColorMap(
      np.uint8(255 * heatmap_resized),
      cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
      np.array(image),
      0.6,
      heatmap_colored,
      0.4,
      0
    )

    st.image(overlay, caption="Grad-CAM Visualization", width=400)

  except Exception  as e:
    st.error(f"Grad-CAM failed: {e}")