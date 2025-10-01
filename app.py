import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# ---- Load TFLite model ----
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---- Load labels ----
with open("plant_disease.json","r") as f:
    labels = json.load(f)

# ---- Streamlit UI ----
st.title("🌱 Plant Disease Detection (TFLite)")
st.write("Upload a leaf image to predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0).astype(np.float32)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred_class = labels[int(np.argmax(pred))]
    confidence = np.max(pred)*100

    st.success(f"Predicted Disease: **{pred_class}**")
    st.info(f"Confidence: {confidence:.2f}%")
