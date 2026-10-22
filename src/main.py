import queue
import threading
from pathlib import Path

import cv2

from mouth_extraction.face_landmarker_extraction import FaceLandmarkerExtractor
from mouth_extraction.mouth_extraction import MouthExtractor
from stabilizer.stabilizer import SentenceStabilizer
from vsr import AutoVSR

BASE_DIR = Path(__file__).resolve().parent.parent


def main(source=0):
    queue_capture = queue.Queue()
    queue_process = queue.Queue()
    queue_predict = queue.Queue()

    stop_recording = False

    extractor = MouthExtractor(FaceLandmarkerExtractor())
    model = AutoVSR(
        str(BASE_DIR / "src" / "vsr" / "models" / "vsr_trlrs2lrs3vox2avsp_base.pth"),
        device="cpu",
    )
    stabilizer = SentenceStabilizer()

    def read_stream(source, batch_size=60, overlap=15):
        nonlocal stop_recording

        vid = cv2.VideoCapture(source)
        frames = []

        num_frames = 0
        current_start_frame = 0

        while vid.isOpened() and not stop_recording:
            ret, frame = vid.read()

            if not ret:
                break

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
            results = model.predict(cropped_frames, k=5)

            # print(results)

            queue_predict.put((start_frame, results, stop_process))

    def stabilize_predictions():
        stop_predict = False

        while not stop_predict:
            start_frame, results, stop_predict = queue_predict.get()
            # print("Stabilizing predictions from queue_predict")
            current_phrase = stabilizer(start_frame, results)
            # print(current_phrase)

    read_thread = threading.Thread(target=read_stream, args=(source,))
    process_thread = threading.Thread(target=process_frames)
    predict_thread = threading.Thread(target=predict_frames)
    stabilize_thread = threading.Thread(target=stabilize_predictions)

    read_thread.start()
    process_thread.start()
    predict_thread.start()
    stabilize_thread.start()

    if source == 0:
        input("Press Enter to stop recording...")
        stop_recording = True

    read_thread.join()
    process_thread.join()
    predict_thread.join()
    stabilize_thread.join()

    print("Final stabilized phrase:", stabilizer.get_phrases_text())


if __name__ == "__main__":
    main()
    # main(str(BASE_DIR / "inputs" / "Sans titre.mp4"))
