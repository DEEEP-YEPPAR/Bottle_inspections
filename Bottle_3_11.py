import sys
import os
import streamlit as st
import tempfile
import numpy as np
from ultralytics import YOLO
import torch

# Check if we're in a headless environment
HEADLESS = "DISPLAY" not in os.environ

try:
    if HEADLESS:
        # Use headless version of OpenCV
        import cv2
        from cv2 import cv2
    else:
        import cv2
except ImportError:
    st.error("OpenCV not installed. Trying to fallback to headless version.")
    try:
        import cv2
        from cv2 import cv2
    except ImportError as e:
        st.error(f"Critical Error: OpenCV could not be loaded. {e}")
        st.stop()

# Verify environment versions
st.info(f"OpenCV version: {cv2.__version__}")
st.info(f"Python version: {sys.version}")
st.info(f"PyTorch version: {torch.__version__}")

# Initialize YOLO model
@st.cache_resource
def load_model():
    try:
        # Proper YOLOv8 model loading for Python 3.11
        model_path = "Weights/Weights/3000_best_60single.pt"
        
        # Verify model path exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {os.path.abspath(model_path)}")
            return None
            
        model = YOLO(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Make sure model path is correct and file exists")
        return None

model = load_model()
class_names = {0: 'defect', 1: 'bottle', 2: 'bottle_neck'}
MM_PER_PIXEL = 0.12  # Calibration factor

# Streamlit UI
st.title("🍾 Bottle Inspection System")
st.markdown("Upload a video file or use live webcam feed")

# Input source selection
input_source = st.radio("Select input source:", ("Upload Video", "Webcam"), 
                        horizontal=True, index=0)

# Initialize video capture
cap = None
tfile = None

if input_source == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()  # Close the file so OpenCV can access it
        cap = cv2.VideoCapture(tfile.name)
        st.info(f"Loaded video: {uploaded_file.name}")

elif input_source == "Webcam":
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)  # Default webcam
        if not cap.isOpened():
            st.error("Could not open webcam")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            st.info("Webcam started. Press 'Stop Processing' to end.")

# Placeholder for video display
frame_placeholder = st.empty()
stop_button = st.button("Stop Processing")

if model is None:
    st.stop()

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
line_height = 25

# Process video frames
if cap and cap.isOpened():
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            if input_source == "Upload Video":
                st.warning("End of video reached")
            else:
                st.warning("Unable to read frame from webcam")
            break
        
        # YOLO prediction
        try:
            # Proper inference call for YOLOv8
            results = model.predict(frame, conf=0.2, verbose=False)
            neck_measurements = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, cls, conf in zip(boxes, classes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        class_id = int(cls)
                        class_name = class_names.get(class_id, 'unknown')
                        confidence = float(conf)
                        
                        # Set colors
                        color = (0, 0, 255)  # Red for defects
                        label_lines = [f"{class_name} {confidence:.2f}"]
                        
                        if class_name == "bottle":
                            color = (255, 0, 0)  # Blue
                        elif class_name == "bottle_neck":
                            color = (0, 255, 0)  # Green
                            neck_width = (x2 - x1) * MM_PER_PIXEL
                            neck_height = (y2 - y1) * MM_PER_PIXEL
                            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                            neck_measurements.append((neck_width, neck_height, centroid))
                            label_lines.append(f"Width: {neck_width:.1f}mm")
                            label_lines.append(f"Height: {neck_height:.1f}mm")
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label background
                        if label_lines:
                            total_height = line_height * len(label_lines)
                            text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] 
                                         for line in label_lines]
                            max_line_width = max(size[0] for size in text_sizes)
                            
                            label_y = max(y1 - total_height, 10)
                            cv2.rectangle(
                                frame, 
                                (x1, label_y), 
                                (x1 + max_line_width + 10, label_y + total_height), 
                                color, 
                                -1
                            )
                            
                            # Draw label text
                            for i, line in enumerate(label_lines):
                                y = label_y + (i + 1) * line_height - 7
                                cv2.putText(frame, line, (x1 + 5, y), 
                                            font, font_scale, (255, 255, 255), font_thickness)
                        
                        # Draw centroid for bottle neck
                        if class_name == "bottle_neck":
                            cv2.drawMarker(frame, centroid, color, 
                                          markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        
        # Prepare frame for Streamlit
        try:
            # Convert BGR to RGB for proper display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, use_column_width=True)
        except Exception as e:
            st.error(f"Display error: {str(e)}")
    
    # Release resources
    try:
        cap.release()
    except:
        pass
    
    if tfile and os.path.exists(tfile.name):
        try:
            os.unlink(tfile.name)
        except:
            pass

st.info("Processing stopped.")
