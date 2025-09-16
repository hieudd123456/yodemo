import os
import tempfile
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit as st
from ultralytics import YOLO, solutions
import av
from ultralytics.utils.plotting import Annotator, colors

# Load the YOLOv8 model for Object Detection
model = YOLO("yolov8n.pt")

# Function to process each frame of the video stream
def process_frame(frame):
    # Read image from the frame with PyAV
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(img, tracker="bytetrack.yaml")

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Return the annotated frame
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Function to process the video with OpenCV
def process_video(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, tracker="bytetrack.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()

# Function to count objects in the video
def count_objects(video_file):
    # Define region points
    region_points = [(20, 150), (400, 150), (400, 350), (20, 350)]

    # Init Object Counter
    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=region_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            tracks = model.track(frame, persist=True, show=False)

            # Count objects in the frame
            annotated_frame = counter.start_counting(frame, tracks)

            # Display the annotated frame
            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object
    cap.release()

# Function to crop objects in the video
def crop_objects(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    with frame_placeholder.container():
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame
                results = model.predict(frame, show=False)

                # Retrieve the bounding boxes and class labels
                boxes = results[0].boxes.xyxy.cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                # Create an Annotator object for drawing bounding boxes
                annotator = Annotator(frame, line_width=2, example=model.names)

                # If boxes are detected, crop the objects and save them to a directory
                if boxes is not None:
                    # Iterate over the detected boxes and class labels
                    for box, cls in zip(boxes, clss):
                        # Draw the bounding box on the frame
                        annotator.box_label(box, color=colors(int(cls), True), label=model.names[int(cls)])

                        # Crop the object from the frame
                        annotated_frame = frame[int(box[1]): int(box[3]), int(box[0]): int(box[2])]

                        # Display the annotated frame
                        if annotated_frame.shape[0] > 0 and annotated_frame.shape[1] > 0:
                            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            st.image(frame, channels="RGB")
            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture object
    cap.release()

# Function to blur objects in the video
def blur_objects(video_file):
    # Blur ratio
    blur_ratio = 50

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame
            results = model.predict(frame, show=False)

            # Retrieve the bounding boxes and class labels
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            # Create an Annotator object for drawing bounding boxes
            annotator = Annotator(frame, line_width=2, example=model.names)

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    annotator.box_label(box, color=colors(int(cls), True), label=model.names[int(cls)])

                    obj = frame[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                    blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

                    frame[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = blur_obj

                    # Display the annotated frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame, channels="RGB")
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object
    cap.release()

# Create a WebRTC video streamer with the process_frame callback
webrtc_streamer(key="streamer", video_frame_callback=process_frame, sendback_audio=False,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
               )

# File upload for uploading the video file, placeholder for displaying the frames and button to start processing
file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
button = st.button("Process Video")
button_count = st.button("Count Objects")
button_crop = st.button("Crop Objects")
button_blur = st.button("Blur Objects")
frame_placeholder = st.empty()
# If the button is clicked and a file is uploaded, save the file to a temporary directory and process the video
if button:
    # Save the file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getvalue())
    # Process the video
    process_video(path)

# If the button is clicked and a file is uploaded, save the file to a temporary directory and count the objects
if button_count:
    # Save the file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getvalue())
    # Count objects in the video
    count_objects(path)

# If the button is clicked and a file is uploaded, save the file to a temporary directory and crop the objects
if button_crop:
    # Save the file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getvalue())
    # Count objects in the video
    crop_objects(path)

# If the button is clicked and a file is uploaded, save the file to a temporary directory and blur the objects
if button_blur:
    # Save the file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getvalue())
    # Count objects in the video
    blur_objects(path)
