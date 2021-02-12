import cv2
import numpy as np

def rotate(images, angle, center=None):
    if center is None:
        center = (images[0].shape[1]//2, images[0].shape[0]//2)

    (h, w) = images[0].shape[:2]
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    M_inv = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # perform the actual rotation and return the images
    return [cv2.warpAffine(image, M, (nW, nH)) for image in images]