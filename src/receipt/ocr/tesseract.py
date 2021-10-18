import cv2
import numpy as np
import pytesseract

import config
from receipt.utils.rotate import rotate

import time
import easyocr
reader = easyocr.Reader(['en'], gpu=True)

def image_word_to_string(text_image):
    text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2RGB)
    
    t0 = time.time()
    pred = easy(text_image)
    #pred = tess(text_image)
    t1 = time.time()

    print(f"OCR recognize: '{pred}' ({(t1 - t0) * 1000:.3f} ms)")

    return pred

def easy(text_image):
    global reader

    cv2.imwrite("./text_image.png", text_image)

    # [([[0, 0], [278, 0], [278, 68], [0, 68]], 'A06400036', 0.869632544366335)]
    #print(reader.character)
    predictions = reader.recognize(text_image, decoder='greedy') #, beamWidth=25, adjust_contrast=0.9, contrast_ths=0.1)

    print(predictions)

    result_text = '\n'.join([pred[1] for pred in predictions])
    return result_text

def tess(text_image):
    tesseract_single_word_config = '--psm 13 --oem 1'
    pred = pytesseract.image_to_string(text_image, config=tesseract_single_word_config)
    return pred