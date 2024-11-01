import cv2
import numpy as np


def load_depth_map(path, width: int = 1024, height: int = 1024):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)

    return data.reshape((width, height)).copy()


def load_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
