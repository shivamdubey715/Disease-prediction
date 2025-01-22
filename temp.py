import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load your trained VGG16 model
model = load_model("model.h5")

# Define the class labels
labels = {0: "Uninfected", 1: "Parasitized"}

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize image to the input size of the model
    image = image.resize((50, 50))  # Update this size based on your model input
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Malaria Detection System")
st.write("Upload a cell smear image to classify it as **infected** or **non-infected**.")

# Upload file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    confidence = np.max(prediction) * 100

    # Display result
    st.write(f"**Prediction:** {labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

