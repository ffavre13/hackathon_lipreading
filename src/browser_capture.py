import queue

import cv2


class BrowserCapture:
    def __init__(self, fps=25):
        self.frames = queue.Queue(maxsize=10)
        self.fps = fps
        self.opened = True

    def push(self, frame):
        if self.frames.full():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
        self.frames.put_nowait(frame)

    def isOpened(self):
        return self.opened

    def read(self):
        try:
            return True, self.frames.get(timeout=1)
        except queue.Empty:
            return self.opened, None

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0

    def release(self):
        self.opened = False
