from pathlib import Path

import cv2
import numpy as np
import torch

from mouth_extraction.face_landmarker_extraction import FaceLandmarkerExtractor
from mouth_extraction.mouth_extraction import MouthExtractor
from vsr import AutoVSR

BASE_DIR = Path(__file__).resolve().parent.parent


def main():

    video_path = BASE_DIR / "inputs" / "Green Screen Sample Panasonic HMC150.mp4"

    vid = cv2.VideoCapture(str(video_path))
    frames = []

    while vid.isOpened():
        ret, frame = vid.read()

        if not ret:
            break

        frames.append(frame)

    vid.release()

    cropped_frames = MouthExtractor(FaceLandmarkerExtractor())(frames)

    video = torch.from_numpy(np.stack(cropped_frames)).unsqueeze(1).float()
    video = (video / 255.0 - 0.421) / 0.165

    model = AutoVSR("./vsr_models/vsr_trlrs3_base.pth", device="cpu")

    # Tenseur fourni par votre pipeline existante : (T, 1, 88, 88).uv
    # video = torch.load("clip_prepare.pt", map_location="cpu", weights_only=True)
    results = model.predict(video, k=5)

    for result in results:
        print(result["rank"], result["text"], result["score"])

    ##### TODO : PARTIE A AJOUTER POUR GERER LES RESULTATS


if __name__ == "__main__":
    main()
