import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

# Initialize client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["API_KEY"]
)

# UI
st.set_page_config(page_title="Roboflow Workflow Inference", layout="centered")
st.title("🔍 Roboflow Workflow Inference")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save image temporarily to disk for SDK compatibility
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Running Roboflow workflow..."):
        result = client.run_workflow(
            workspace_name="cranberry-counting-ycg5p",
            workflow_id="small-object-detection-sahi",
            images={"image": temp_path},
            use_cache=True
        )

    st.success("Inference Complete!")
    st.json(result)

    # Optionally display image from result
    if "image" in result and "url" in result["image"]:
        st.image(result["image"]["url"], caption="Inference Result")

    os.remove(temp_path)
