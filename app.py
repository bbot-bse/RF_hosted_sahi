import streamlit as st
import requests
from PIL import Image

# Replace with your Roboflow API Key and Model Endpoint
API_KEY = "9916rXo9gEB3PSlqx1JH"
MODEL_ENDPOINT = "flower-counter/1"  # e.g., "username/project/1"

st.set_page_config(page_title="Roboflow Inference", layout="centered")
st.title("🤖 Roboflow Inference App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image_bytes = uploaded_file.read()

    with st.spinner("Running inference..."):
        response = requests.post(
            f"https://detect.roboflow.com/{MODEL_ENDPOINT}?api_key={API_KEY}",
            files={"file": image_bytes},
        )

    if response.status_code == 200:
        result = response.json()
        image_url = result["image"]["url"]
        st.image(image_url, caption="Predicted", use_column_width=True)
        st.json(result)
    else:
        st.error(f"Error: {response.status_code}")
