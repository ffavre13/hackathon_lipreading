import os
import queue
import threading
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from ollama import Client

from mouth_extraction.face_landmarker_extraction import FaceLandmarkerExtractor
from mouth_extraction.mouth_extraction import MouthExtractor
from stabilizer.stabilizer import SentenceStabilizer
from vsr import AutoVSR

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("OLLAMA_API")

try:
    client = Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + str(api_key)},
    )
except Exception as e:
    print(f"Error creating Ollama clienty: {e}")

process_times = []
predict_times = []
stabilize_times = []


def main(source=0, on_phrase=print, on_frame=None, stop_event=None):
    stop_event = stop_event or threading.Event()

    queue_capture = queue.Queue(maxsize=2)
    queue_process = queue.Queue(maxsize=2)
    queue_predict = queue.Queue(maxsize=2)

    LLM_BATCH_THRESHOLD = 10

    extractor = MouthExtractor(FaceLandmarkerExtractor())
    model = AutoVSR(
        str(BASE_DIR / "src" / "vsr" / "models" / "vsr_trlrs2lrs3vox2avsp_base.pth"),
        device="cpu",
    )
    stabilizer = SentenceStabilizer()

    llm_stabilized_thread = None

    def read_stream(source, batch_seconds=3, overlap_seconds=0.5):
        # vid = cv2.VideoCapture(source)
        vid = source if hasattr(source, "read") else cv2.VideoCapture(source)

        fps = vid.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            fps = 25.0

        extractor.set_fps(fps)

        batch_size = max(15, round(batch_seconds * fps))
        overlap = min(round(overlap_seconds * fps), batch_size - 1)

        frames = []

        num_frames = 0
        current_start_frame = 0

        while vid.isOpened() and not stop_event.is_set():
            ret, frame = vid.read()

            if not ret:
                break

            if frame is None:
                continue

            if on_frame:
                on_frame(frame)

            frames.append(frame)

            num_frames += 1

            if len(frames) == batch_size:
                # print("Sending batch of frames to queue_capture")
                queue_capture.put((current_start_frame, frames.copy(), False))
                frames = frames[batch_size - overlap :]
                current_start_frame = num_frames - overlap

        vid.release()
        # Start time, frame, is_last_batch
        queue_capture.put((current_start_frame, frames.copy(), True))

    def process_frames():
        stop_capture = False

        while not stop_capture:
            start_frame, frames, stop_capture = queue_capture.get()
            start_time = time.time()
            # print("Cropping frames from queue_capture")
            cropped_frames = extractor(frames)

            if len(cropped_frames) == 0 and not stop_capture:
                continue

            process_times.append(time.time() - start_time)
            queue_process.put((start_frame, cropped_frames, stop_capture))

    def predict_frames():
        stop_process = False

        while not stop_process:
            start_frame, cropped_frames, stop_process = queue_process.get()
            start_time = time.time()
            # print("Predicting on cropped frames from queue_process")

            if len(cropped_frames) == 0:
                queue_predict.put((start_frame, None, stop_process))
                continue

            results = model.predict(cropped_frames, k=5)
            predict_times.append(time.time() - start_time)

            # print(results)
            queue_predict.put((start_frame, results, stop_process))

    def stabilize_predictions():
        nonlocal llm_stabilized_thread

        stop_predict = False

        while not stop_predict:
            start_frame, results, stop_predict = queue_predict.get()
            # print("Stabilizing predictions from queue_predict")

            if results is not None:
                start_time = time.time()
                current_phrase = stabilizer(start_frame, results)
                on_phrase(stabilizer.get_phrases_text())
                stabilize_times.append(time.time() - start_time)

            llm_thread_busy = (
                llm_stabilized_thread is not None and llm_stabilized_thread.is_alive()
            )
            if (
                not llm_thread_busy
                and len(stabilizer.stabilized_phrases) % LLM_BATCH_THRESHOLD == 0
            ):
                llm_stabilized_thread = threading.Thread(
                    target=stabilizer.llm_stabilize,
                    args=(client,),
                )
                llm_stabilized_thread.start()

    read_thread = threading.Thread(target=read_stream, args=(source,))
    process_thread = threading.Thread(target=process_frames)
    predict_thread = threading.Thread(target=predict_frames)
    stabilize_thread = threading.Thread(target=stabilize_predictions)

    read_thread.start()
    process_thread.start()
    predict_thread.start()
    stabilize_thread.start()

    read_thread.join()
    process_thread.join()
    predict_thread.join()
    stabilize_thread.join()

    if llm_stabilized_thread is not None:
        llm_stabilized_thread.join()

    llm_stabilized_thread = threading.Thread(
        target=stabilizer.llm_stabilize,
        args=(client, True),
    )
    llm_stabilized_thread.start()
    llm_stabilized_thread.join()

    on_phrase(stabilizer.get_phrases_text())


if __name__ == "__main__":
    stop = threading.Event()
    # t = threading.Thread(
    #     target=main,
    #     kwargs={"source": str(BASE_DIR / "inputs" / "test.mp4"), "stop_event": stop},
    # )
    t = threading.Thread(target=main, kwargs={"stop_event": stop})
    t.start()
    input("Press Enter to stop recording...")
    stop.set()
    t.join()
    
    plot_time = False
    
    if plot_time:

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        plt.title("Process Times Histogram")
        sns.histplot(process_times, kde=True)
        plt.axvline(np.mean(process_times), color="red", linestyle="--")
        plt.text(
            0.97,
            0.95,
            f"Mean = {np.mean(process_times):.2f} s",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="red",
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(str(BASE_DIR / "assets" / "process_times_histogram.png"))

        plt.figure(figsize=(10, 6))
        plt.title("Predict Times Histogram")
        sns.histplot(predict_times, kde=True)
        plt.axvline(np.mean(predict_times), color="red", linestyle="--")
        plt.text(
            0.97,
            0.95,
            f"Mean = {np.mean(predict_times):.2f} s",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="red",
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(str(BASE_DIR / "assets" / "predict_times_histogram.png"))

        plt.figure(figsize=(10, 6))
        plt.title("Stabilize Times Histogram")
        sns.histplot(stabilize_times, kde=True)
        plt.axvline(np.mean(stabilize_times), color="red", linestyle="--")
        plt.text(
            0.97,
            0.95,
            f"Mean = {np.mean(stabilize_times):.2f} s",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="red",
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(str(BASE_DIR / "assets" / "stabilize_times_histogram.png"))
