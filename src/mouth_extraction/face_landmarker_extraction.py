import mediapipe as mp
import cv2

class FaceLandmarkerExtractor:
    def __init__(self, model_path = "./models/face_landmarker.task"):
        self.model_path = model_path
        self.base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        self.face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(base_options=self.base_options)
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(self.face_landmarker_options)
        
    def extract_landmarks(self, frames):
        face_landmarkers = []
        
        for frame in frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = self.face_landmarker.detect(mp_image)
            face_landmarkers.append(result) if result else face_landmarkers.append(None)
            
        return face_landmarkers

    def draw_landmarks(self, frames, landmarks):
        for frame, result in zip(frames, landmarks):
            if result is None:
                continue

            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        return frames