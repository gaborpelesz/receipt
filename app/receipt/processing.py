import cv2
import numpy as np
import time
import pytesseract
import re

import config
from utils.rotate import rotate

class Receipt():
    def __init__(self, cropped_receipt):
        self.date, self.AP = "?", "?"
        self.cropped_receipt = cropped_receipt

    def _post_process_AP(self, AP_code):
        if len(AP_code) < 8:
            return AP_code
        print('before post process:', AP_code)
        last_digit_idx = re.match('.+([0-9])[^0-9]*$', AP_code).start(1)

        post_processed = AP_code[last_digit_idx-7:last_digit_idx+1]
        post_processed = post_processed.replace('Q', '0')
        post_processed = post_processed.replace('A', '4')
        post_processed = post_processed.replace('G', '6')

        return 'A' + post_processed

    def _post_process_date(self, date):
        post_processed = date
        post_processed.replace(',', '.')
        return date

    def get_date_and_AP(self):

        image = self.cropped_receipt[3*self.cropped_receipt.shape[0]//4:, :]
        image_height, image_width = image.shape[:2]

        text_images = self.find_text(image)

        if len(text_images) == 0:
            return "?", "?"

        AP_candidate = text_images[0][0]
        date_candidate = None

        for text_image in text_images[1:]:
            if text_image[1][0] < image_width//3:
                date_candidate = text_image[0]
                break

        ocr_start = time.time()
        self.AP = self._post_process_AP(image_word_to_string(AP_candidate))

        if date_candidate is not None:
            self.date = self._post_process_date(image_word_to_string(date_candidate))
        ocr_end = time.time()
        print(f'     AP code: {self.AP}')
        print(f'        date: {self.date}')
        
        print(f'ocr runtime: {(ocr_end-ocr_start)*1000:.2f}ms') # time
        
        return self.date, self.AP

    def get_all_text(self):
        img = self.cropped_receipt[3*self.cropped_receipt.shape[0]//4:, :]

        t_start = time.time()
        ocr_text = pytesseract.image_to_string(img)
        print(f'{(time.time()-t_start)*1000:.3f} ms')

        with open('test.txt', 'w') as f:
            f.write(ocr_text)

    def get_text(self, text_images):
        image = text_images

        # t_start = time.time()
        config = '--psm 8 --oem 1'
        ocr_text = pytesseract.image_to_string(image, config=config)
        # print(f'OCR: {(time.time()-t_start)*1000:.3f} ms')

        with open('test.txt', 'w') as f:
            f.write(ocr_text)

    def find_text(self, image):
        im_height, im_width = image.shape[:2]


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)


        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 2))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mask = np.zeros(bw.shape, dtype=np.uint8)

        text_boxes = []
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y+h, x:x+w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

            if r > 0.30 and w/im_width > 0.08 and h > 16:
                rect = cv2.minAreaRect(contours[idx])

                text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

                convex_hull = cv2.convexHull(contours[idx])
                cv2.drawContours(text_mask, [convex_hull], -1, 255, -1)
                smoothed_mask = cv2.dilate(text_mask, kernel=np.ones((4,4),dtype=np.uint8), iterations=4)
                text_boxes.append((rect, smoothed_mask))
  

        text_images = []
        smoothed_image = None
        for text_box in text_boxes:
            angle = -text_box[0][2] if text_box[0][2] > -45 else 270-text_box[0][2]

            rotated_image, rotated_mask = rotate([image, text_box[1]], angle, center=text_box[0][0])

            contour = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            x, y, w, h = cv2.boundingRect(contour)

            rotated_mask = rotated_mask[y:y+h,x:x+w]
            rotated_mask = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)
            smoothed_image = cv2.bitwise_and(rotated_image[y:y+h,x:x+w], rotated_mask)
            smoothed_image = cv2.add(smoothed_image, cv2.bitwise_not(rotated_mask))
            
            position = (text_box[0][0][0], text_box[0][0][1]) # text center position on the image

            text_images.append((smoothed_image, position))


        if config.DEBUG:
            for i, text_box in enumerate(text_boxes):
                box = cv2.boxPoints(text_box[0])
                box = np.int0(box)

                image = cv2.putText(image, f'{i+1}', tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.drawContours(image,[box],0,(0,0,255),2)

            for i, text_image in enumerate(text_images[:10]):
                cv2.namedWindow(f'text image {i+1}', cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f'text image {i+1}', 1200,1000)
                cv2.imshow(f'text image {i+1}', text_image[0])

            cv2.namedWindow('text boxes', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('text boxes', 1200,1000)
            cv2.imshow('text boxes', self.cropped_receipt)

            cv2.namedWindow('grad', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('grad', 1200,1000)
            cv2.imshow('grad', grad)

            cv2.namedWindow('otzu', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('otzu', 1200,1000)
            cv2.imshow('otzu', bw)

            cv2.namedWindow('connected', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('connected', 1200,1000)
            cv2.imshow('connected', connected)
        return text_images

def smooth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 1)
    return adaptive

def image_word_to_string(text_image):
    tesseract_single_word_config = '--psm 8 --oem 1'
    return pytesseract.image_to_string(text_image, config=tesseract_single_word_config)