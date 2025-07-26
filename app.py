# streamlit_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
model = tf.keras.models.load_model("hand2.keras")

st.title("🖊️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) in the canvas below:")

canvas_result = st.canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    image = canvas_result.image_data
    image = Image.fromarray((image[:, :, 0]).astype(np.uint8))  # extract one channel
    image = ImageOps.invert(image)  # white on black
    image = image.resize((28, 28)).convert("L")
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    st.image(image.resize((140, 140)), caption="Input Digit", use_column_width=False)

    pred = model.predict(img_array)
    st.write(f"### Predicted Digit: {np.argmax(pred)}")

