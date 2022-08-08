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
st.markdown("<h1 style='text-align: center;'>😷Mask Recognition Application</h1>",
            unsafe_allow_html=True)

st.markdown("<h4>🖼  Detect From Image</h4>", unsafe_allow_html=True)

img = st.file_uploader("Choose a file")
col1, col2 = st.columns(2)
if img is not None:
    img_trans = Image.open(img)
    predict = m.Predict()
    pred, pred_prob = predict.predict(img_trans)
    with col1:
        st.image(img)
    with col2:
        st.markdown(
            f"<h4> Recognition : {pred}</h4>  ", unsafe_allow_html=True)
        st.markdown(
            f"<h4> Recognition Probs : {pred_prob}</h4>  ", unsafe_allow_html=True)


st.markdown("<h4> 📸 Detect From Camera</h4>", unsafe_allow_html=True)

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

st.markdown("<h3> What is this ❓</h3 > <p> Different from object detection, object recognition here <b> only predicts what the object is, so if there are many objects, this model will not work well </b>, Therefore, Object recognition is usually used in certain cases where only 1 object needs to be detected, such as the detection of cancer or covid. In this application, I use the Algorithm MobileNet, you can read the paper at the following link:: <a href = 'https://arxiv.org/abs/1801.04381' >🔗MobileNetv2 Paper🔗 </a> </p>", unsafe_allow_html=True)

st.markdown("<h3> Resource I used 🧰 </h3> <p> MobileNetv2 : <a href = 'https://pytorch.org/hub/pytorch_vision_mobilenet_v2/' > Pytorch Hub</a></p> <p> Mask Dataset : Google Image Search", unsafe_allow_html=True)
