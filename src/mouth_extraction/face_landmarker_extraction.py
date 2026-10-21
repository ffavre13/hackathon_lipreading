from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models" / "face_landmarker.task"


class FaceLandmarkerExtractor:
    def __init__(self, model_path=str(MODELS_PATH)):
        self.model_path = model_path
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_model_path(
            self.model_path
        )

    def __call__(self, frames):
        return self.extract_landmarks(frames)

    def extract_landmarks(self, frames):
        face_landmarkers = []

        for frame in frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self.face_landmarker.detect(mp_image)

            if not result.face_landmarks:
                face_landmarkers.append(None)
                continue

            # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
            LEFT_EYE_IDX = [33, 133]
            RIGHT_EYE_IDX = [263, 362]
            NOSE_IDX = 1
            MIDDLE_TOP_LIP_IDX = 13
            MIDDLE_BOTTOM_LIP_IDX = 14

            # Convert normalized landmarks to pixel coordinates
            face_pts = np.array(
                [
                    [pt.x * frame.shape[1], pt.y * frame.shape[0]]
                    for pt in result.face_landmarks[0]
                ]
            )

            # Extract key landmarks for the face (eyes, nose and mouth center points for the crop and the mouth corners and the middle top and bottom lip points for checking if the mouth is open )
            key_landmarks = np.array(
                [
                    face_pts[LEFT_EYE_IDX].mean(0),
                    face_pts[RIGHT_EYE_IDX].mean(0),
                    face_pts[NOSE_IDX],
                    face_pts[[MIDDLE_TOP_LIP_IDX, MIDDLE_BOTTOM_LIP_IDX]].mean(0),
                    face_pts[MIDDLE_TOP_LIP_IDX],
                    face_pts[MIDDLE_BOTTOM_LIP_IDX],
                ]
            )

            face_landmarkers.append(key_landmarks)

        return face_landmarkers

    def draw_landmarks(self, frames, landmarks):
        for frame, result in zip(frames, landmarks):
            if result is None:
                continue

            for points in result:
                x = int(points[0])
                y = int(points[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        return frames
