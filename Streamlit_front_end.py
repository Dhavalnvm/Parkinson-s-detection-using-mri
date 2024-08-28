import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model_path = r"C:\Users\Acer\Parkinsons\model.h5" # Update with your actual model path if different
model = tf.keras.models.load_model(model_path)

# Streamlit app
st.title("Image Classification with VGG16")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = load_img(uploaded_file, target_size=(224, 224))  # Resize to 224x224
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make predictions
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        st.write("The image is classified as: Parkinson's Disease")
    else:
        st.write("The image is classified as: Normal")

# Add footer or any additional information
st.write("Upload an image to classify it using the pre-trained model.")
