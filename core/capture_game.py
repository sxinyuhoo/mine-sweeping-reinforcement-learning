
import cv2
import numpy as np
from mss import mss

def capture_screen(region=None):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Example usage:
if __name__ == "__main__":
    region = {'top': 100, 'left': 100, 'width': 800, 'height': 600}
    img = capture_screen(region)
    cv2.imshow('Screenshot', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()