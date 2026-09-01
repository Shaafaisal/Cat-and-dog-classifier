import streamlit as st
import tensorflow as tf  # type: ignore
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

model = tf.keras.models.load_model("cat_dog_model.keras")

st.title("🐱 Cat vs Dog Classifier")

uploaded_file = st.file_uploader(
    "Upload a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    # Resize to match training size
    image = image.resize((160, 160))

    image_array = np.array(image)
    
    # Ensure 3 channels (RGB)
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[:, :, :3]
    
    image_array = np.expand_dims(image_array, axis=0)
    
    # Apply preprocessing (important for MobileNetV2)
    image_array = preprocess_input(image_array)

    prediction = model.predict(image_array)[0][0]

    if prediction > 0.5:
        st.success(f"🐶 Dog detected! Confidence: {prediction * 100:.2f}%")
    else:
        st.success(f"🐱 Cat detected! Confidence: {(1 - prediction) * 100:.2f}%")