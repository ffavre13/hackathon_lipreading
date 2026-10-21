import os

import cv2
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

st.title("Lip reading")

tab1, tab2, tab3 = st.tabs(["Camera", "Video", "Youtube"])

with tab1:
    st.write("Camera")

    run = st.toggle("Activate camera", value=False)
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    counter = 1
    t = st.empty()

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        t.write("Counter: " + str(counter))

        counter += 1


with tab2:
    st.write("Video")

    save_path = os.getcwd() + "/inputs"

    file = st.file_uploader("Upload video .mp4", type=["mp4"])
    file_path = save_path

    if file is not None:
        file_path = f"{save_path}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success("File uploaded successfully!")

    if os.path.isfile(file_path):
        with open(file_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

with tab3:
    st.write("Youtube")

    youtube_url = st.text_input("Enter Youtube URL")
    if youtube_url:
        st.video(youtube_url)

        video_id = youtube_url.split("v=")[-1]
        video_transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

        st.write(video_transcript)
