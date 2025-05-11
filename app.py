import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import time
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox
import matplotlib.pyplot as plt


# Streamlit page configuration
st.set_page_config(page_title="Wildfire Detection App", page_icon="🔥", layout="wide")

# Configure Sentinel Hub (replace with your credentials)
config = SHConfig()
config.instance_id = '2e3c4b35-7c67-467f-ab88-c416e0d19cbd'  # Replace with your instance ID
config.sh_client_id = '5c56e12c-1fcd-4ec1-9d64-4d23bd2da231'   # Replace with your client ID
config.sh_client_secret = 'Wlr967gPsiq21UkSGCfqEUSDIc2RX5HQ'  # Replace with your client secret

# If credentials are not set, use local images as fallback
USE_SENTINEL_HUB = bool(config.instance_id and config.sh_client_id and config.sh_client_secret)

# Define constants
IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'wildfire_model_final.h5')
SAMPLE_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_images')

# Load the trained model with error handling
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the 'models' directory contains 'wildfire_model_final.h5'.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Preprocess image for model
def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    if img_array.shape[-1] != 3:  # Ensure RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict wildfire presence
def predict_wildfire(image, model):
    try:
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]
        label = "Wildfire" if prediction > 0.5 else "No Wildfire"
        return label, prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Fetch Sentinel-2 image
def get_sentinel_image():
    try:
        # Define area of interest (e.g., a region prone to wildfires in California)
        bbox = BBox(bbox=[-122.5, 39.0, -122.0, 39.5], crs=CRS.WGS84)  # [min_x, min_y, max_x, max_y]
        
        # Define Sentinel-2 request
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B03", "B02"],
                output: { bands: 3 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
        """
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L1C)],
            responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
            bbox=bbox,
            size=[512, 512],
            config=config
        )
        
        response = request.get_data()[0]
        return Image.fromarray(response)
    except Exception as e:
        st.warning(f"Error fetching Sentinel-2 image: {e}")
        return None

# Get sample image from local directory (fallback)
def get_sample_image():
    if not os.path.exists(SAMPLE_IMAGE_DIR):
        st.warning(f"Sample image directory {SAMPLE_IMAGE_DIR} not found.")
        return None
    images = [f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        st.warning("No sample images found.")
        return None
    img_path = os.path.join(SAMPLE_IMAGE_DIR, np.random.choice(images))
    return Image.open(img_path)

# Streamlit UI
st.title("Wildfire Detection from Satellite Imagery")
st.markdown("""
This app uses a trained deep learning model to detect wildfires in satellite images.
- **Live Feed**: Simulates a live satellite feed using Sentinel-2 images or local samples.
- **Upload Image**: Upload your own image for wildfire detection.
- **Model**: Trained on satellite imagery to classify wildfire vs. no wildfire.
""")

# Tabs for live feed and upload
tab1, tab2 = st.tabs(["Live Satellite Feed", "Upload Image"])

with tab1:
    st.header("Live Satellite Feed")
    st.markdown("Displays recent satellite images and checks for wildfires. Updates every 30 seconds.")
    
    # Placeholder for image and prediction
    image_placeholder = st.empty()
    prediction_placeholder = st.empty()
    
    # Simulate live feed
    if st.button("Start Live Feed", key="start_feed"):
        while True:
            if USE_SENTINEL_HUB:
                image = get_sentinel_image()
            else:
                image = get_sample_image()
            
            if
                prediction_placeholder.error("Unable to fetch image. Please try again or upload an image.")
                break
            
            # Display image
            image_placeholder.image(image, caption="Current Satellite Image", use_container_width=True)
            
            # Predict
            label, prob = predict_wildfire(image, model)
            if label and prob is not None:
                prediction_placeholder.markdown(
                    f"**Prediction**: {label} (Confidence: {prob:.4f})",
                    unsafe_allow_html=True
                )
            else:
                prediction_placeholder.error("Prediction failed.")
            
            # Wait before next update
            time.sleep(30)  # Update every 30 seconds
            # Check for stop button
            if st.button("Stop Live Feed", key="stop_feed"):
                break

with tab2:
    st.header("Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict
        label, prob = predict_wildfire(image, model)
        if label and prob is not None:
            st.markdown(f"**Prediction**: {label} (Confidence: {prob:.4f})")
        else:
            st.error("Prediction failed.")

# Footer
st.markdown("""
---
**Note**: The live feed is simulated using Sentinel-2 images or local samples due to limited access to real-time satellite streams. For production use, consider paid APIs like Planet Labs.
**Model**: Trained on the wildfire prediction dataset. Ensure images are clear and relevant for accurate predictions.
**Source**: Sentinel Hub for satellite imagery.
**Setup**: Ensure the 'models' directory contains 'wildfire_model_final.h5' and Sentinel Hub credentials are configured.
""")