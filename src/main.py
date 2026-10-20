import cv2
from mouth_extraction.face_landmarker_extraction import FaceLandmarkerExtractor

def main():
    video_path = "./src/mouth_extraction/data/test.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        frames.append(frame)

    cap.release()

    extractor = FaceLandmarkerExtractor("./src/mouth_extraction/models/face_landmarker.task")
    landmarks  = extractor.extract_landmarks(frames)
    
    frames_with_landmarks = extractor.draw_landmarks(frames, landmarks)
    
    print(landmarks[0])

if __name__ == "__main__":
    main()