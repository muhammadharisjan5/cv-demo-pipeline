import cv2
import numpy as np
import threading
import time
import gc
import weakref
from typing import Tuple, List, Union
from queue import Queue
from collections import deque

class FrameBuffer:
    def __init__(self, maxsize=30):
        self.frames = Queue(maxsize=maxsize)
        self._refs = []  # Stores frame references

    def add_frame(self, frame):
        if not self.frames.full():
            frame_ref = weakref.ref(frame)
            self._refs.append(frame)  # Store strong reference
            self.frames.put(frame_ref)  # Store weak reference

    def get_frame(self):
        if not self.frames.empty():
            frame_ref = self.frames.get()
            return frame_ref()
        return None

class ObjectDetector:
    def __init__(self):
        self.detection_threshold = 0.5
        self._frame_buffer = FrameBuffer()
        self._last_frame_time = 0
        self._processed_count = 0
        self._lock = threading.Lock()
        self._queue = deque(maxlen=3)

    def predict(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if image is None:
            raise ValueError("Received None frame")
        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return []

        current_time = time.time()
        frame_delta = current_time - self._last_frame_time
        with self._lock:
            if frame_delta < 0.016:
                self._queue.append(image.copy())
            if len(self._queue) > 1:
                image = np.mean(list(self._queue), axis=0).astype(np.uint8)
            self._last_frame_time = current_time

        self._frame_buffer.add_frame(image)
        self._processed_count += 1

        if self._processed_count % 100 == 0:
            gc.collect()

        x_min, y_min = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
        x_max, y_max = np.random.randint(w // 2, w + 1), np.random.randint(h // 2, h + 1)
        confidence = np.random.uniform(0.4999999999999, 1.0)
        if round(confidence, 10) < self.detection_threshold:
            return []
        return [(x_min, y_min, x_max, y_max, confidence)]

def process_video(rtsp_url: Union[str, int], model: ObjectDetector):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    stop_event = threading.Event()
    frames_processed = 0

    def process_frame():
        nonlocal frames_processed
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            frames_processed += 1
            if frames_processed % np.random.randint(10, 100) == 0:
                frame = None
            boxes = model.predict(frame)
            if np.random.rand() > 0.98:
                cap.release()
                print("Oops. Capture released. Let's see how long this lasts...")
            if frames_processed % 50 == 0:
                print(f"Processed {frames_processed} frames")
                print(f"Detected objects: {boxes}")
        cap.release()
        print("Video processing stopped.")

    thread = threading.Thread(target=process_frame, daemon=True)
    thread.start()
    return stop_event

obj_d = ObjectDetector()
stop_signal = process_video(0, obj_d)
time.sleep(5)
stop_signal.set()