import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('plant_disease_recog_model_pwp.keras')

# Define class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Tomato___Late_blight', 'Tomato___Early_blight']  # add all classes

# Streamlit UI
st.title("🌱 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=['jpg','jpeg','png'])
if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption='Uploaded Image')
    
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)
    
    pred = model.predict(x)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)*100
    
    st.success(f"Predicted Disease: {pred_class}")
    st.info(f"Confidence: {confidence:.2f}%")
