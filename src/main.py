from multiprocessing import Value
import cv2
import numpy as np
from mouth_extraction.face_landmarker_extraction import FaceLandmarkerExtractor
import torch
from vsr import AutoVSR


def main():
    video_path = "./src/mouth_extraction/data/test.mp4"

    ##### TODO: PARTIE A REMPLACER PAR L'OUTPUT DE MEDIAPIPE 
    ##### Il faudra sûrement gérer la boucle (window glissante ici et tout prendre dedans)
    cap = cv2.VideoCapture(video_path)
    frames = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (88, 88), interpolation=cv2.INTER_AREA)
            frames.append(gray)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No valid frames in the video : {video_path}")

    video = torch.from_numpy(np.stack(frames)).unsqueeze(1).float()
    video = (video / 255.0 - 0.421) / 0.165

    ###### FIN DE LA PARTIE A REMPLACER ----------------------------------------------

    model = AutoVSR("./vsr_models/vsr_trlrs3_base.pth", device="cpu")

    # Tenseur fourni par votre pipeline existante : (T, 1, 88, 88).
    # video = torch.load("clip_prepare.pt", map_location="cpu", weights_only=True)
    results = model.predict(video, k=5)

    for result in results:
        print(result["rank"], result["text"], result["score"])


    ##### TODO : PARTIE A AJOUTER POUR GERER LES RESULTATS
    
if __name__ == "__main__":
    main()
