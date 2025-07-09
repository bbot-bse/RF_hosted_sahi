import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

# Securely read API key
API_KEY = st.secrets["API_KEY"]

# Initialize the Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# Streamlit UI
st.set_page_config(page_title="Roboflow Workflow Inference", layout="centered")
st.title("🔍 Roboflow Workflow Inference")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Running Roboflow workflow..."):
        try:
            result = client.run_workflow(
                workspace_name="cranberry-counting-ycg5p",
                workflow_id="small-object-detection-sahi",
                images={"image": temp_path},
                use_cache=True
            )
            st.success("Inference Complete!")
            st.json(result)

            # Display output image if available
            if "image" in result and "url" in result["image"]:
                st.image(result["image"]["url"], caption="Inference Result", use_container_width=True)

        except Exception as e:
            st.error(f"Workflow failed: {e}")

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)