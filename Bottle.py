


import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import os
import sys
import torch


# Verify OpenCV installation
st.info(f"OpenCV version: {cv2.__version__}")
st.info(f"Python version: {sys.version}")

# Initialize YOLO model
# @st.cache_resource
# def load_model():
#     try:
#         model = YOLO("Weights//Weights//3000_best_60single.pt")
#         st.success("Model loaded successfully!")
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None
@st.cache_resource
def load_model():
    import torch
    from packaging import version

    try:
        if version.parse(torch.__version__) >= version.parse("2.6"):
            from torch.serialization import add_safe_globals
            from ultralytics.nn.tasks import DetectionModel
            add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

        # Load YOLO model (this works with both old and new PyTorch)
        model = YOLO("Weights//Weights//3000_best_60single.pt")
        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

        

model = load_model()
class_names = {0: 'defect', 1: 'bottle', 2: 'bottle_neck'}
MM_PER_PIXEL = 0.12  # Calibration factor

# Streamlit UI
st.title("🍾 Bottle Inspection System")
st.markdown("Upload a video file or use live webcam feed")

# Input source selection
input_source = st.radio("Select input source:", ("Upload Video", "Webcam"), horizontal=True)

# Initialize video capture
cap = None
tfile = None

if input_source == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

elif input_source == "Webcam":
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)  # Default webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
    st_frame = st.empty()
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video or unable to read frame.")
            break
        
        # YOLO prediction
        try:
            results = model.predict(frame, conf=0.2, verbose=False)
            neck_measurements = []
            
            for result in results:
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
                    total_height = line_height * len(label_lines)
                    max_line_width = max(
                        cv2.getTextSize(line, font, font_scale, font_thickness)[0][0]
                        for line in label_lines
                    )
                    label_y = max(y1 - total_height, 0)
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
                        cv2.putText(frame, line, (x1 + 5, y), font, font_scale, (255, 255, 255), font_thickness)
                    
                    # Draw centroid for bottle neck
                    if class_name == "bottle_neck":
                        cv2.drawMarker(frame, centroid, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        
        # Prepare frame for Streamlit
        try:
            # Display frame
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        except Exception as e:
            st.error(f"Display error: {str(e)}")
    
    # Release resources
    cap.release()
    if tfile:
        os.unlink(tfile.name)  # Clean up temp file

st.info("Processing stopped.")





# # import streamlit as st
# # import cv2
# # import tempfile
# # from ultralytics import YOLO
# # import numpy as np
# # import os

# # # Initialize YOLO model
# # @st.cache_resource
# # def load_model():
# #     # Update the path to your model weights (use relative path for portability)
# #     return YOLO("Weights//3000_best_60single.pt")  # Adjust path as needed

# # model = load_model()
# # class_names = {0: 'defect', 1: 'bottle', 2: 'bottle_neck'}
# # MM_PER_PIXEL = 0.12  # Calibration factor

# # # Streamlit UI
# # st.title("Bottle Inspection System")
# # st.markdown("Upload a video file or use live webcam feed")

# # # Input source selection
# # input_source = st.radio("Select input source:", ("Upload Video", "Webcam"))

# # # Initialize video capture
# # cap = None

# # if input_source == "Upload Video":
# #     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
# #     if uploaded_file:
# #         tfile = tempfile.NamedTemporaryFile(delete=False)
# #         tfile.write(uploaded_file.read())
# #         cap = cv2.VideoCapture(tfile.name)

# # elif input_source == "Webcam":
# #     if st.button("Start Webcam"):
# #         cap = cv2.VideoCapture(0)  # Default webcam
# #         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# #         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # # Placeholder for video display
# # frame_placeholder = st.empty()
# # stop_button = st.button("Stop")

# # # Font settings
# # font = cv2.FONT_HERSHEY_SIMPLEX
# # font_scale = 0.8
# # font_thickness = 2
# # line_height = 25

# # # Process video frames
# # if cap and cap.isOpened():
# #     while cap.isOpened() and not stop_button:
# #         ret, frame = cap.read()
# #         if not ret:
# #             st.warning("End of video or unable to read frame.")
# #             break
        
# #         # YOLO prediction
# #         results = model.predict(frame, conf=0.2, verbose=False)
# #         neck_measurements = []
        
# #         for result in results:
# #             boxes = result.boxes.xyxy.cpu().numpy()
# #             classes = result.boxes.cls.cpu().numpy()
# #             confidences = result.boxes.conf.cpu().numpy()
            
# #             for box, cls, conf in zip(boxes, classes, confidences):
# #                 x1, y1, x2, y2 = map(int, box)
# #                 class_id = int(cls)
# #                 class_name = class_names.get(class_id, 'unknown')
# #                 confidence = float(conf)
                
# #                 # Set colors
# #                 color = (0, 0, 255)  # Red for defects
# #                 label_lines = [f"{class_name} {confidence:.2f}"]
                
# #                 if class_name == "bottle":
# #                     color = (255, 0, 0)  # Blue
# #                 elif class_name == "bottle_neck":
# #                     color = (0, 255, 0)  # Green
# #                     neck_width = (x2 - x1) * MM_PER_PIXEL
# #                     neck_height = (y2 - y1) * MM_PER_PIXEL
# #                     centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
# #                     neck_measurements.append((neck_width, neck_height, centroid))
# #                     label_lines.append(f"Width: {neck_width:.1f}mm Height: {neck_height:.1f}mm")
                
# #                 # Draw bounding box
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
# #                 # Draw label background
# #                 total_height = line_height * len(label_lines)
# #                 max_line_width = max(
# #                     cv2.getTextSize(line, font, font_scale, font_thickness)[0][0]
# #                     for line in label_lines
# #                 )
# #                 label_y = max(y1 - total_height, 0)
# #                 cv2.rectangle(
# #                     frame, 
# #                     (x1, label_y), 
# #                     (x1 + max_line_width + 10, label_y + total_height), 
# #                     color, 
# #                     -1
# #                 )
                
# #                 # Draw label text
# #                 for i, line in enumerate(label_lines):
# #                     y = label_y + (i + 1) * line_height - 7
# #                     cv2.putText(frame, line, (x1 + 5, y), font, font_scale, (255, 255, 255), font_thickness)
                
# #                 # Draw centroid for bottle neck
# #                 if class_name == "bottle_neck":
# #                     cv2.drawMarker(frame, centroid, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
# #         # Prepare frame for Streamlit
# #         rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
# #         resized = cv2.resize(rotated, (600, 800))
# #         rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
# #         # Display frame
# #         frame_placeholder.image(rgb_frame, channels="RGB")
        
# #         # Slow down processing for smoother display
# #         cv2.waitKey(10)
    
# #     # Release resources
# #     cap.release()
# #     cv2.destroyAllWindows()
# #     if input_source == "Upload Video" and uploaded_file:
# #         os.unlink(tfile.name)  # Clean up temp file

# # st.info("Stopped video processing.")
