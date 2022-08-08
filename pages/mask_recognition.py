import streamlit as st
import torch
import torchvision
import script.model as m
from PIL import Image
import cv2
import av
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)
st.markdown("<h1 style='text-align: center;'>Mask Recognition Application</h1>",
            unsafe_allow_html=True)

img = st.file_uploader("Choose a file")

if img is not None:
    img_trans = Image.open(img)
    predict = m.Predict()
    pred, pred_prob = predict.predict(img_trans)
    st.header("Machine Prediction : " + pred)
    st.write("Machine Confident : ")
    st.write(pred_prob)
    st.image(img)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


st.write("Refresh halaman atau ulangi klik tombol start jika camera tidak muncul")
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 40)
fontScale = 1
fontColor = (50, 168, 82)
thickness = 2
lineType = 1


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_trans = Image.fromarray(img)
    pred, pred_prob = m.Predict().predict(img_trans)
    if pred == "with mask":
        fontColor = (12, 92, 7)
    elif pred == "without mask":
        fontColor = (247, 42, 35)
    cv2.putText(img, pred + " " + str(pred_prob),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="opencv-filter",
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
