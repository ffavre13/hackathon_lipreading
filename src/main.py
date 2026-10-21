import queue
import threading
from pathlib import Path

import cv2

from mouth_extraction.face_landmarker_extraction import FaceLandmarkerExtractor
from mouth_extraction.mouth_extraction import MouthExtractor
from vsr import AutoVSR

BASE_DIR = Path(__file__).resolve().parent.parent


def main(source=0):
    queue_capture = queue.Queue()
    queue_process = queue.Queue()
    queue_predict = queue.Queue()

    stop_recording = False

    extractor = MouthExtractor(FaceLandmarkerExtractor())
    model = AutoVSR("./vsr_models/vsr_trlrs3_base.pth", device="cpu")

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
            next_start_frame, frames, stop_capture = queue_capture.get()
            # print("Cropping frames from queue_capture")
            cropped_frames = extractor(frames)
            
            if len(cropped_frames) == 0 and not stop_capture:
                continue

            queue_process.put((next_start_frame, cropped_frames, stop_capture))

    def predict_frames():
        stop_process = False

        while not stop_process:
            next_start_frame, cropped_frames, stop_process = queue_process.get()
            # print("Predicting on cropped frames from queue_process")
            results = model.predict(cropped_frames, k=5)

            print(results[0]["text"])

            queue_predict.put((next_start_frame, results, stop_process))

    read_thread = threading.Thread(target=read_stream, args=(source,))
    process_thread = threading.Thread(target=process_frames)
    predict_thread = threading.Thread(target=predict_frames)

    read_thread.start()
    process_thread.start()
    predict_thread.start()

    if source == 0:
        input("Press Enter to stop recording...")
        stop_recording = True

    read_thread.join()
    process_thread.join()
    predict_thread.join()

    ##### TODO : PARTIE A AJOUTER POUR GERER LES RESULTATS


if __name__ == "__main__":
    main()
    # main(str(BASE_DIR / "inputs" / "Sans titre.mp4"))
