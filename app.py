import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# -----------------------------
# Load disease info from JSON
# -----------------------------
with open("plant_disease.json", "r") as f:
    disease_info = json.load(f)

# Create a list of class names for predictions
class_names = [item["name"] for item in disease_info]

# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="plant_disease_model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the disease and get treatment information.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Adjust if your model has a different input size
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make prediction using TFLite interpreter
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class and confidence
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    # Get cause and cure from JSON
    cause = disease_info[predicted_index]["cause"]
    cure = disease_info[predicted_index]["cure"]

    # Display results
    st.markdown(f"### Predicted Disease: {predicted_class}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown(f"**Cause:** {cause}")
    st.markdown(f"**Cure / Treatment:** {cure}")

    # Optional: Show top-3 predictions
    top_indices = prediction[0].argsort()[-3:][::-1]
    st.subheader("Top-3 Predictions:")
    for i in top_indices:
        name = class_names[i]
        conf = prediction[0][i]*100
        st.write(f"{name}: {conf:.2f}%")
