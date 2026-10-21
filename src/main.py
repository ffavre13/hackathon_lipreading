from pathlib import Path

import cv2

from mouth_extraction.face_landmarker_extraction import FaceLandmarkerExtractor
from mouth_extraction.mouth_extraction import MouthExtractor

BASE_DIR = Path(__file__).resolve().parent


def main():
    video_path = BASE_DIR / "mouth_extraction" / "data" / "test.mp4"
    
    vid = cv2.VideoCapture(str(video_path))
    frames = []
    
    while vid.isOpened():
        ret, frame = vid.read()

        if not ret:
            break
        
        frames.append(frame)

    vid.release()
    
    cropped_frames = MouthExtractor(FaceLandmarkerExtractor())(frames)


if __name__ == "__main__":
    main()
