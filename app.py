import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
from main import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_image():
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    st.info("⚠️ Note:The programs and models are a little out of sink ,please try twice or trice for new analysis and Image adjustmnets.")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)  # Convert PIL image to NumPy array for OpenCV processing
        return image, uploaded_file.name,True
    return None, None,False

def zoom_image(image, scale_factor, pan_x, pan_y):
    """Zoom and scroll (pan) the image based on user input."""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Apply panning
    new_x = int(center_x + pan_x * w)
    new_y = int(center_y + pan_y * h)

    # Compute zoomed dimensions
    new_w = int(w / scale_factor)
    new_h = int(h / scale_factor)

    # Ensure crop area stays within bounds
    x1 = max(0, new_x - new_w // 2)
    x2 = min(w, x1 + new_w)
    y1 = max(0, new_y - new_h // 2)
    y2 = min(h, y1 + new_h)

    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (w, h))  # Resize back to original shape

    return resized

def adjust_brightness_contrast(image, brightness, contrast):
    """Apply brightness and contrast adjustments."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def Main():
    st.set_page_config(layout="wide")

    col1, col2, col3 = st.columns([1, 3, 1])

    # Load image
    load = False
    image, image_name,load = load_image()

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        with col1:
            st.header("Image Info")
            st.text(f"Size: {image_pil.size}")
            st.text(f"shape: {image.shape}")
            st.text(f"Mode: {image_pil.mode}")

        with col2:
            subcol1, subcol2 = st.columns(2)

            with subcol1:
                st.header("Original Image")
                st.image(image, use_container_width=True)

            # Store user settings in session state
            if "zoom" not in st.session_state:
                st.session_state.zoom = 1.0
            if "brightness" not in st.session_state:
                st.session_state.brightness = 0
            if "contrast" not in st.session_state:
                st.session_state.contrast = 1.0
            if "pan_x" not in st.session_state:
                st.session_state.pan_x = 0.0
            if "pan_y" not in st.session_state:
                st.session_state.pan_y = 0.0

            # Apply transformations
            # image = cv2.resize(image, (544, 544), interpolation=cv2.INTER_AREA)
            new = False
            if len(image.shape) ==2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            img_res,image_info = Model.predict(image, image_name,new)
            img_transformed = adjust_brightness_contrast(img_res, st.session_state.brightness, st.session_state.contrast)
            img_transformed = zoom_image(img_transformed, st.session_state.zoom, st.session_state.pan_x, st.session_state.pan_y)

            with subcol2:
                if load:
                    loading_placeholder = st.empty()
                    animations = ["🔄", "⏳", "⌛", "🕐", "🔃"]
                    duration = 3
                    start_time = time.time()
                    while time.time() - start_time < duration:
                        for icon in animations:
                            loading_placeholder.markdown(f"### {icon} Loading...")
                            time.sleep(0.5)  # Adjust for smooth animation
                    loading_placeholder.empty()  # Clear after loading
                    load=False
                st.header("possible fractures")
                st.image(img_transformed, use_container_width =True)
                if image_info["is_prev_data"]:
                    if image_info["date and time"]:
                        st.text(f"previous data date: {image_info['date and time']}")
                    st.text("want a new analysis")
                    if st.button('Yes'):
                        load=True
                        new = True
                        img_res,image_info = Model.predict(image, image_name,new)
                        img_transformed = adjust_brightness_contrast(img_res, st.session_state.brightness, st.session_state.contrast)
                        img_transformed = zoom_image(img_transformed, st.session_state.zoom, st.session_state.pan_x, st.session_state.pan_y)

        with col3:
            st.header("Image Adjustments")

            # Zoom slider
            st.session_state.zoom = st.slider("Zoom", 1.0, 3.0, st.session_state.zoom, 0.1)

            # Brightness and Contrast sliders
            st.session_state.brightness = st.slider("Brightness", -100, 100, st.session_state.brightness)
            st.session_state.contrast = st.slider("Contrast", 0.5, 3.0, st.session_state.contrast)

            # Pan sliders
            st.session_state.pan_x = st.slider("Pan X", -0.5, 0.5, st.session_state.pan_x, 0.01)
            st.session_state.pan_y = st.slider("Pan Y", -0.5, 0.5, st.session_state.pan_y, 0.01)

            # Refresh Button
            if st.button("Reset Image Adjustments"):
                st.session_state.zoom = 1.0
                st.session_state.brightness = 0
                st.session_state.contrast = 1.0
                st.session_state.pan_x = 0.0
                st.session_state.pan_y = 0.0
                st.experimental_rerun()
        

if __name__ == "__main__":
    Main()
