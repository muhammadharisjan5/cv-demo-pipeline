import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from cv_pipeline import ObjectDetector, FrameBuffer, process_video

def test_invalid_image():
    detector = ObjectDetector()
    with pytest.raises(ValueError):
        detector.predict(None)

def test_predict_with_valid_image():
    detector = ObjectDetector()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = detector.predict(image)
    assert isinstance(result, list)
    assert len(result) == 1

@patch("cv2.VideoCapture")
def test_process_video(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (True, np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    mock_video_capture.return_value = mock_capture

    detector = ObjectDetector()
    stop_signal = process_video("fake_url", detector)
    assert stop_signal is not None
