import cv2
import numpy as np
import pytesseract

import config
from receipt.utils.rotate import rotate

def image_word_to_string(text_image):
    tesseract_single_word_config = '--psm 13 --oem 1'
    return pytesseract.image_to_string(text_image, config=tesseract_single_word_config)