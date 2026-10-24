import os
import threading
import time

import streamlit as st

from main import main


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
    run = st.toggle("Activate camera", value=False)
    frame_elem = st.empty()
    st.write("Camera feed:")
    text_elem = st.empty()
    text_elem.write(st.session_state.get("cam_text", ""))

    if run:
        text_elem.write("Waiting for camera...")
        process = start_pipeline(0)
        st.session_state.cam = process

        try:
            display(process, text_elem, frame_elem, save_key="cam_text")
        finally:
            process["stop"].set()
    elif "cam" in st.session_state:
        display(st.session_state.pop("cam"), text_elem, save_key="cam_text")


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
