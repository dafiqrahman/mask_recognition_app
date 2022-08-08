import streamlit as st
import numpy as np
import torch
import torchvision
from script.model_detection import model
from PIL import Image
import cv2
import os
from datetime import datetime
import av
import threading
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    VideoProcessorBase,
    webrtc_streamer,
)


st.markdown("<h1 style='text-align: center;'>😷 Mask Detection Application</h1>",
            unsafe_allow_html=True)


st.markdown("<h4>🖼  Detect From Image</h4><p>Im reccomend detect from image, since using realtime camera is broken when I deploy it on the website.</p>",
            unsafe_allow_html=True)
# ------------ by Image --------------
image_file = st.file_uploader("Choose a file")
col1, col2 = st.columns(2)
if image_file is not None:
    img = Image.open(image_file)
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width='always')
    ts = datetime.timestamp(datetime.now())
    imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
    outputpath = os.path.join(
        'data/outputs', os.path.basename(imgpath))
    with open(imgpath, mode="wb") as f:
        f.write(image_file.getbuffer())

    # call Model prediction--
    pred = model(imgpath)
    pred.render()  # render bbox in image
    for im in pred.imgs:
        im_base64 = Image.fromarray(im)
        im_base64.save(outputpath)

    # --Display predicton

    img_ = Image.open(outputpath)
    with col2:
        st.image(img_, caption='Model Prediction(s)',
                 use_column_width='always')

    # delete temp files
    os.remove(imgpath)
    os.remove(outputpath)

# --------------------- end by image ------------------------

st.markdown("<h4> 📸 Detect From Camera</h4>", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.write("Refresh halaman atau ulangi klik tombol start jika camera tidak muncul")


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.style = 'color'

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)

        # image processing code here

        return av.VideoFrame.from_ndarray(np.squeeze(results.render()), format="bgr24")


webrtc_streamer(
    key="opencv-filter",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


st.markdown("<h3> What is this ❓</h3 > <p> this is an implementation of object detection in computer vision, where the machine looks for patterns to find out <b> what, where, and how many objects it wants to detect </b>. In this application im using an algorithm called YOLO (You Only Look Once) version 5, you can read the paper at the following link: <a href = 'https: // arxiv.org/abs/2108.11539' >🔗YOLOv5 Paper🔗 </a> </p>", unsafe_allow_html=True)

st.markdown("<h3> Resource I used 🧰 </h3> <p> YOLOv5 : <a href = 'https: // github.com/ultralytics/yolov5' > YOLOv5 Github</a></p> <p> Mask Dataset : <a href = 'https://universe.roboflow.com/khdas567-gmail-com/mask-wearing-eblit'> Roboflow </a>", unsafe_allow_html=True)
