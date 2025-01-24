import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load your trained VGG16 model
model = load_model("model1.h5")

# Define the class labels
labels = {0: "Uninfected", 1: "Parasitized"}

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((50, 50))  # Resize to match model input size
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.set_page_config(page_title="Malaria Detection System", page_icon="🩺")

# Sidebar
st.sidebar.title("Malaria Detection System")
st.sidebar.info(
    "Upload a cell smear image to classify it as **infected** or **non-infected**. "
    "The model will predict the likelihood of infection with confidence."
)

# Main title
st.markdown(
    "<h1 style='text-align: center; color: #FF6347;'>Malaria Detection System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>"
    "This AI-powered tool helps in detecting malaria-infected cells from microscopic images. </p>",
    unsafe_allow_html=True,
)

# Upload file
uploaded_file = st.file_uploader(
    "📤 Upload a cell smear image (JPG/PNG/JPEG format):",
    type=["jpg", "png", "jpeg"],
)

if uploaded_file is not None:
    # Display uploaded image
    st.markdown("<h3 style='text-align: center;'>Uploaded Image</h3>", unsafe_allow_html=True)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Add a button to trigger prediction
    if st.button("🔍 Predict"):
        # Make predictions
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        confidence = np.max(prediction) * 100

        # Display result
        st.markdown("<h3 style='text-align: center;'>Prediction Results</h3>", unsafe_allow_html=True)
        st.write(
            f"**Prediction:** {'🟢 Uninfected' if predicted_class == 0 else '🔴 Parasitized'}"
        )
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Progress bar for confidence
        st.progress(confidence / 100)
else:
    st.markdown(
        "<p style='text-align: center; color: gray;'>No image uploaded yet. Please upload an image to proceed.</p>",
        unsafe_allow_html=True,
    )
