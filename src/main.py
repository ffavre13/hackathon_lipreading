import os
import queue
import threading
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

    def read_stream(source, batch_size=75, overlap=15):

        vid = cv2.VideoCapture(source)
        extractor.set_fps(vid.get(cv2.CAP_PROP_FPS))

        frames = []

        num_frames = 0
        current_start_frame = 0

        while vid.isOpened() and not stop_event.is_set():
            ret, frame = vid.read()

            if not ret:
                break

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
            # print("Cropping frames from queue_capture")
            cropped_frames = extractor(frames)

            if len(cropped_frames) == 0 and not stop_capture:
                continue

            queue_process.put((start_frame, cropped_frames, stop_capture))

    def predict_frames():
        stop_process = False

        while not stop_process:
            start_frame, cropped_frames, stop_process = queue_process.get()
            # print("Predicting on cropped frames from queue_process")
            
            if len(cropped_frames) == 0:
                queue_predict.put((start_frame, None, stop_process))
                continue
            
            results = model.predict(cropped_frames, k=5)

            # print(results)

            queue_predict.put((start_frame, results, stop_process))

    def stabilize_predictions():
        nonlocal llm_stabilized_thread

        stop_predict = False

        while not stop_predict:
            start_frame, results, stop_predict = queue_predict.get()
            # print("Stabilizing predictions from queue_predict")
            
            if results is not None:
                current_phrase = stabilizer(start_frame, results)
                on_phrase(stabilizer.get_phrases_text())

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
        args=(client,True),
    )
    llm_stabilized_thread.start()
    llm_stabilized_thread.join()
    
    on_phrase(stabilizer.get_phrases_text())

if __name__ == "__main__":
    stop = threading.Event()
    # t = threading.Thread(target=main, kwargs={"source": str(BASE_DIR / "inputs" / "test.mp4"), "stop_event": stop})
    t = threading.Thread(target=main, kwargs={"stop_event": stop})
    t.start()
    input("Press Enter to stop recording...")
    stop.set()
    t.join()