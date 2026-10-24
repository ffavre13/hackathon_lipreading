from pathlib import Path

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
MEAN_FACE_PATH = BASE_DIR / "models" / "20words_mean_face.npy"


class MouthExtractor:
    # The pipeline for this class is strongly inspired by https://github.com/mpc001/auto_avsr/blob/main/preparation/detectors/mediapipe/video_process.py

    def __init__(
        self,
        face_landmarker_extractor,
        smooth_window_size=15,
        mean_face_path=str(MEAN_FACE_PATH),
        crop_width=88,
        crop_height=88,
        fps=25
    ):
        self.face_landmarker_extractor = face_landmarker_extractor
        self.smooth_window_size = smooth_window_size
        self.mean_face = np.load(mean_face_path)
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.fps = fps
        
    def set_fps(self, fps):
        self.fps = fps

    def __call__(self, frames, last_valid_landmark=None):
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        landmarks = self.face_landmarker_extractor(frames)

        interpolated_landmarks = self.interpolate_landmarks(
            landmarks, last_valid_landmark
        )

        if not interpolated_landmarks:
            return []

        smoothed_landmarks = self.smooth_landmarks(interpolated_landmarks)

        frames_affine_transformed, landmarks_affine_transformed = self.affine_transform(
            frames, smoothed_landmarks
        )

        cropped_mouth_frames = self.crop_mouth(
            frames_affine_transformed, landmarks_affine_transformed
        )        
        
        # for f in cropped_mouth_frames:
        #     cv2.imshow("Cropped Mouth", f)
        #     cv2.waitKey(0)
        
        cropped_mouth_frames = self.remove_silent_frames(cropped_mouth_frames, interpolated_landmarks)
        if not cropped_mouth_frames:
            return []
        
        cropped_mouth_frames = self.resample_frames(cropped_mouth_frames)            
            
        normalized_cropped_mouth_frames = (
            torch.from_numpy(np.stack(cropped_mouth_frames)).unsqueeze(1).float()
        )
        normalized_cropped_mouth_frames = (
            normalized_cropped_mouth_frames / 255.0 - 0.421
        ) / 0.165

        return normalized_cropped_mouth_frames
    
    def remove_silent_frames(self, frames, landmarks, threshold=0.04, min_silence=0.5):
        min_len = round(min_silence * self.fps)

        closed = [np.linalg.norm(l[4] - l[5]) / np.linalg.norm(l[0] - l[1]) < threshold for l in landmarks]

        kept = []
        i = 0
        while i < len(frames):
            j = i
            while j < len(frames) and closed[j] == closed[i]:
                j += 1
            if not (closed[i] and j - i >= min_len):
                kept.extend(frames[i:j])
            i = j
        return kept

    def resample_frames(self, frames, target_fps=25):
        n_out = round(len(frames) * target_fps / self.fps)
        idx = (np.arange(n_out) * self.fps / target_fps).astype(int)
        idx = np.clip(idx, 0, len(frames) - 1)
        return [frames[i] for i in idx]
        

    def interpolate_landmarks(self, landmarks, last_valid_landmark=None):

        valid_landmarks_idx = [
            i for i, landmark in enumerate(landmarks) if landmark is not None
        ]

        if not valid_landmarks_idx:
            return None

        if last_valid_landmark is not None:
            all_landmarks = [last_valid_landmark] + landmarks
            valid_landmarks_idx = [0] + [i + 1 for i in valid_landmarks_idx]
        else:
            all_landmarks = landmarks

        for i in range(1, len(valid_landmarks_idx)):
            if valid_landmarks_idx[i] - valid_landmarks_idx[i - 1] > 1:
                start_idx = valid_landmarks_idx[i - 1]
                stop_idx = valid_landmarks_idx[i]
                delta = all_landmarks[stop_idx] - all_landmarks[start_idx]
                for j in range(start_idx + 1, stop_idx):
                    all_landmarks[j] = all_landmarks[start_idx] + delta * (
                        (j - start_idx) / (stop_idx - start_idx)
                    )

        if last_valid_landmark is not None:
            all_landmarks = all_landmarks[1:]

        # Fill the start and end

        valid_landmarks_idx = [
            i for i, landmark in enumerate(landmarks) if landmark is not None
        ]

        for i in range(valid_landmarks_idx[0]):
            all_landmarks[i] = all_landmarks[valid_landmarks_idx[0]]

        for i in range(valid_landmarks_idx[-1] + 1, len(all_landmarks)):
            all_landmarks[i] = all_landmarks[valid_landmarks_idx[-1]]

        return all_landmarks

    def smooth_landmarks(self, landmarks):
        smoothed_landmarks = []
        half_window = self.smooth_window_size // 2

        for i in range(len(landmarks)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(landmarks), i + half_window + 1)

            window_landmarks = [
                landmark
                for landmark in landmarks[start_idx:end_idx]
                if landmark is not None
            ]

            if window_landmarks:
                smoothed_landmark = np.mean(window_landmarks, axis=0)
                smoothed_landmark += landmarks[i].mean(0) - smoothed_landmark.mean(0)
                smoothed_landmarks.append(smoothed_landmark)
            else:
                smoothed_landmarks.append(landmarks[i])

        return smoothed_landmarks

    def affine_transform(self, frames, landmarks):
        stable_reference = np.vstack(
            [
                np.mean(self.mean_face[36:42], axis=0),
                np.mean(self.mean_face[42:48], axis=0),
                np.mean(self.mean_face[31:36], axis=0),
                np.mean(self.mean_face[48:68], axis=0),
            ]
        )
        reference_width = 256
        reference_height = 256

        new_frames = []
        new_landmarks = []

        for frame, landmark in zip(frames, landmarks):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            M = cv2.estimateAffinePartial2D(
                landmark[:4], stable_reference, method=cv2.LMEDS
            )[0]

            new_frame = cv2.warpAffine(frame, M, (reference_width, reference_height))
            new_landmark = cv2.transform(landmark.reshape(-1, 1, 2), M).reshape(-1, 2)

            new_frames.append(new_frame)
            new_landmarks.append(new_landmark)

        return new_frames, new_landmarks

    def crop_mouth(self, frames, landmarks):
        cropped_mouth_frames = []

        for frame, landmark in zip(frames, landmarks):
            center_x, center_y = landmark[3]

            y_min = round(center_y - self.crop_height // 2)
            y_max = y_min + self.crop_height
            x_min = round(center_x - self.crop_width // 2)
            x_max = x_min + self.crop_width

            cropped_frame = frame[y_min:y_max, x_min:x_max]
            cropped_mouth_frames.append(cropped_frame)

        return cropped_mouth_frames
