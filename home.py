import streamlit as st
import cv2
import artifact.model as m
from PIL import Image
import torch
from streamlit_webrtc import webrtc_streamer
import av

st.markdown("<h1 style='text-align: center;'>Mask Recognition Application</h1>",
            unsafe_allow_html=True)


st.write("bingung isi pembukaan apaan, langsung ke use camera atau ga image(klik panah pojok kiri atas)")
