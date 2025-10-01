import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# -----------------------------
# Load disease info from JSON
# -----------------------------
@st.cache_data
def load_disease_info(json_path="plant_disease.json"):
    with open(json_path, "r") as f:
        disease_info = json.load(f)
    class_names = [item["name"] for item in disease_info]
    return disease_info, class_names

disease_info, class_names = load_disease_info()

# -----------------------------
# Load TFLite model
# -----------------------------
@st.cache_resource
def load_tflite_model(model_path="plant_disease_model_quant.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the disease and get treatment information.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # Adjust if your model input size is different
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict using TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction results
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100
    cause = disease_info[predicted_index]["cause"]
    cure = disease_info[predicted_index]["cure"]

    # Display results
    st.markdown(f"### Predicted Disease: {predicted_class}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown(f"**Cause:** {cause}")
    st.markdown(f"**Cure / Treatment:** {cure}")

    # Show top-3 predictions
    top_indices = prediction[0].argsort()[-3:][::-1]
    st.subheader("Top-3 Predictions:")
    for i in top_indices:
        st.write(f"{class_names[i]}: {prediction[0][i]*100:.2f}%")
