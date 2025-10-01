import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# ---- Load quantized TFLite model ----
interpreter = tf.lite.Interpreter(model_path="plant_disease_model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---- Load plant disease info JSON ----
with open("plant_disease.json","r") as f:
    plant_diseases = json.load(f)

# ---- Streamlit UI ----
st.title("🌱 Plant Disease Detection (TFLite Quantized)")
st.write("Upload a leaf image to predict its disease and get cause & cure.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess for TFLite model
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred_index = int(np.argmax(pred))
    confidence = np.max(pred)*100

    # Get disease info from JSON
    disease_info = plant_diseases[pred_index]
    disease_name = disease_info["name"]
    disease_cause = disease_info["cause"]
    disease_cure = disease_info["cure"]

    # Display results
    st.success(f"Predicted Disease: **{disease_name}**")
    st.info(f"Confidence: {confidence:.2f}%")
    st.write(f"**Cause:** {disease_cause}")
    st.write(f"**Cure/Management:** {disease_cure}")
