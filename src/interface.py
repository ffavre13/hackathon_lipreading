import os
import threading
import time

import streamlit as st

from main import main

from streamlit_webrtc import webrtc_streamer

from browser_capture import BrowserCapture

def start_pipeline(source):
    latest = {"frame": None, "phrase": None}
    stop = threading.Event()
    thread = threading.Thread(
        target=main,
        kwargs=dict(
            source=source,
            on_phrase=lambda p: latest.update(phrase=p),
            on_frame=lambda f: latest.update(frame=f),
            stop_event=stop,
        ),
        daemon=True,
    )
    thread.start()
    return {"thread": thread, "latest": latest, "stop": stop}


def display(process, text_elem, frame_elem=None, save_key=None):
    while process["thread"].is_alive():
        phrase = process["latest"]["phrase"]
        if save_key and phrase:
            st.session_state[save_key] = phrase
        if frame_elem is not None and process["latest"]["frame"] is not None:
            frame_elem.image(process["latest"]["frame"], channels="BGR")
        if phrase is not None:
            text_elem.write(phrase)
        time.sleep(0.05)

    phrase = process["latest"]["phrase"] or ""
    text_elem.write(phrase)


st.title("Lip reading")

tab1, tab2 = st.tabs(["Camera", "Video"])

with tab1:
    if "capture" not in st.session_state:
        st.session_state.capture = BrowserCapture()
    capture = st.session_state.capture
      
    def on_video_frame(frame):
        capture.push(frame.to_ndarray(format="bgr24"))
        return frame

    ctx = webrtc_streamer(
        key="cam",
        video_frame_callback=on_video_frame,
        media_stream_constraints={"video": True, "audio": False},
    )

    text_elem = st.empty()
    text_elem.write(st.session_state.get("cam_text", ""))

    if ctx.state.playing:
        capture.opened = True
        text_elem.write("Waiting for camera...")
        st.session_state.process = start_pipeline(capture)
        process = st.session_state.process
        try:
            while ctx.state.playing and process["thread"].is_alive():
                phrase = process["latest"]["phrase"]
                if phrase:
                    st.session_state.cam_text = phrase
                    text_elem.write(phrase)
                time.sleep(0.05)
        finally:
            capture.release()
            process["stop"].set()
            
    elif "process" in st.session_state:
        process = st.session_state.pop("process")
        while process["thread"].is_alive():
            phrase = process["latest"]["phrase"]
            if phrase:
                st.session_state.cam_text = phrase
                text_elem.write(phrase)
            time.sleep(0.05)
        
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

    st.write("Lip reading result:")
    text_elem = st.empty()

    if os.path.isfile(file_path) and st.button("Execute lip reading", key="video"):
        display(start_pipeline(file_path), text_elem)
