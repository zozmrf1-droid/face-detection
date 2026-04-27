import streamlit as st
import cv2
import dlib
import time
import numpy as np
import os
from PIL import Image

# --- Configurations ---
OPENCV_HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
DLIB_MODEL_PATH = 'mmod_human_face_detector.dat'

# 1. --- UI Setup ---
st.set_page_config(page_title="AI Face Detector", layout="centered", page_icon="🤖")

st.title("🤖 AI Face Detector")
st.write("Upload an image and use Artificial Intelligence to detect faces! Compare standard OpenCV against Deep Learning with dlib.")

# Sidebar Configuration
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("Choose AI Algorithm", ("OpenCV (Haar Cascade)", "dlib (Deep Learning)", "Run Both"))

# File Uploader
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Image Processing Prep ---
    # Convert uploaded file to a numpy array for OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Check if image loaded correctly
    if image is None:
        st.error("Error reading the image file.")
        st.stop()
        
    st.markdown("### Results")
    
    # We create two columns to show results side by side if running Both
    col1, col2 = st.columns(2) if mode == "Run Both" else (st.columns(1)[0], None)
    
    # --- OPENCV ALGORITHM ---
    if mode in ["OpenCV (Haar Cascade)", "Run Both"]:
        with col1 if mode == "Run Both" else st.container():
            st.subheader("OpenCV Result")
            img_cv = image.copy()
            # Grayscale is required for Haar cascades
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(OPENCV_HAAR_PATH)
            
            start_time = time.time()
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            end_time = time.time()
            
            # Draw boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 3) # Blue box
                
            # Convert BGR back to RGB for Streamlit to render colors correctly
            img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            st.image(img_cv_rgb, channels="RGB")
            st.success(f"OpenCV found {len(faces)} face(s) in {end_time - start_time:.4f} seconds.")

    # --- DLIB ALGORITHM ---
    if mode in ["dlib (Deep Learning)", "Run Both"]:
        with col2 if mode == "Run Both" else st.container():
            st.subheader("dlib CNN Result")
            if not os.path.exists(DLIB_MODEL_PATH):
                st.error("Error: Could not find dlib .dat file. Make sure it's in the project folder!")
            else:
                img_dlib = image.copy()
                cnn_face_detector = dlib.cnn_face_detection_model_v1(DLIB_MODEL_PATH)
                
                # dlib specifically wants RGB
                img_dlib_rgb = cv2.cvtColor(img_dlib, cv2.COLOR_BGR2RGB)
                
                start_time = time.time()
                faces = cnn_face_detector(img_dlib_rgb, 1)
                end_time = time.time()
                
                # Draw boxes
                for face in faces:
                    rect = face.rect
                    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                    cv2.rectangle(img_dlib_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3) # Green box
                    
                st.image(img_dlib_rgb, channels="RGB")
                st.success(f"dlib CNN found {len(faces)} face(s) in {end_time - start_time:.4f} seconds.")
