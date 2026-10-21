import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python#live-stream_1
    """)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():

    import cv2
    import mediapipe as mp

    return cv2, mp


@app.cell
def _():
    model_path = "./models/face_landmarker.task"
    return (model_path,)


@app.cell
def _(cv2):
    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (cam,)


@app.cell
def _(cam, cv2, model_path, mp):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )

    timestamp_ms = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cam.read()

            # OpenCV donne du BGR ; MediaPipe attend du RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            cv2.imshow(
                "Camera",
                rgb_frame,
            )

            if cv2.waitKey(1) == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run()
