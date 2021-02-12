import time
import re
import cv2
import numpy as np

import config
from receipt.ocr import tesseract as ocr
from receipt.text_detection import detector

import craft_text_detector as craft

class Receipt():
    def __init__(self, cropped_receipt):
        self.date, self.AP = "?", "?"
        self.cropped_receipt = cropped_receipt

        self.runtime_ocr = 0
        self.runtime_findtext = 0
        self.text_images = None
        self.found_text_boxes = None

        self.raw_AP = None
        self.raw_date = None
        self.raw_time = None

        self.AP_text_box = None
        self.time_text_box = None
        self.date_text_box = None

    def _post_process_AP(self, AP_code):
        if len(AP_code) < 8:
            return AP_code
        
        post_processed = AP_code
        post_processed = post_processed.replace('d', '0')
        post_processed = post_processed.replace('D', '0')
        post_processed = post_processed.replace('g', '0')
        post_processed = post_processed.replace('M', '0')
        post_processed = post_processed.replace('N', '0')
        post_processed = post_processed.replace('O', '0')
        post_processed = post_processed.replace('p', '0')
        post_processed = post_processed.replace('Q', '0')
        post_processed = post_processed.replace('V', '0')
        post_processed = post_processed.replace('#', '0')
        post_processed = post_processed.replace('I', '1')
        post_processed = post_processed.replace('i', '1')
        post_processed = post_processed.replace('L', '1')
        post_processed = post_processed.replace('T', '1')
        post_processed = post_processed.replace('?', '1')
        post_processed = post_processed.replace('B', '3')
        post_processed = post_processed.replace('A', '4')
        post_processed = post_processed.replace('S', '5')
        post_processed = post_processed.replace('§', '5')
        post_processed = post_processed.replace('$', '5')
        post_processed = post_processed.replace('b', '6')
        post_processed = post_processed.replace('e', '6')
        post_processed = post_processed.replace('E', '6')
        post_processed = post_processed.replace('G', '6')
        post_processed = post_processed.replace('K', '6')
        post_processed = post_processed.replace(' ', '')

        match = re.match('.+([0-9])[^0-9]*$', post_processed)

        if match is None:
            return post_processed

        last_digit_idx = match.start(1)

        post_processed = post_processed[last_digit_idx-7:last_digit_idx+1]

        # dealing with common ocr errors
        # If we know the vendors then we can check which AP codes are relevant for them
        #   https://mkeh.gov.hu/hivatal/kozerdeku_adatok_2013/tevekenysegre_mukodesre_vonatkozo_adatok/nyilvantartasok
        # on the above website under the "Pénztárgép forgalmazási engedélyek jegyzéke" download the PDF
        if post_processed[:3] == '060': post_processed = '064' + post_processed[3:] # 064 for ROSSMANN, 060 for MOL
        if post_processed[:3] == '034': post_processed = '064' + post_processed[3:] # there isn't any machine with 034 AP

        return 'A' + post_processed

    def _post_process_date(self, date):
        post_processed = date

        post_processed = post_processed.replace('Q', '0')
        post_processed = post_processed.replace('A', '4')
        post_processed = post_processed.replace('G', '6')
        post_processed = post_processed.replace('P', '2') # TODO check

        date_candidate = re.findall('([0-9]{0,2})', post_processed)

        if date_candidate is None:
            return post_processed

        date_candidate = [i for i in date_candidate if i] # remove empty strings

        if len(date_candidate) < 3:
            return post_processed

        date_candidate = date_candidate[-3:]

        #  month postprocess
        if len(date_candidate[1]) == 2:
            # example: '92' -> '02'
            if date_candidate[1][0] == '9': date_candidate[1] = '0' + date_candidate[1][1]

        # if found then the date will be in form: [year, month, day] where 'year' has only the last two digits
        # assuming that this application will not be used in 2100<
        return f'20{date_candidate[0]}.{date_candidate[1]}.{date_candidate[2]}.'

    def _post_process_time(self, time):
        post_processed = time

        post_processed = post_processed.replace('O', '0')
        post_processed = post_processed.replace('o', '0')
        post_processed = post_processed.replace('{', '1')

        time_candidate = re.findall('([0-9]{0,2})', post_processed)

        if time_candidate is None:
            return post_processed

        time_candidate = [i for i in time_candidate if i] # remove empty strings

        if len(time_candidate) < 2:
            return post_processed

        # hour post process
        if len(time_candidate[0]) == 2:
            # example: '44' -> '14'
            if time_candidate[0][0] == '4': time_candidate[0] = '1' + time_candidate[0][1]

        # minute post process
        if len(time_candidate[1]) == 2:
            # example: '94' -> '34'
            if time_candidate[1][0] == '9': time_candidate[1] = '3' + time_candidate[1][1]
            if time_candidate[1][0] == '6': time_candidate[1] = '5' + time_candidate[1][1]

        return f'{time_candidate[0]}:{time_candidate[1]}'

    def process(self):
        # last 1/4 of the receipt is enough to be processed for AP, date and time
        image = self.cropped_receipt[3*self.cropped_receipt.shape[0]//4:, :]
        image_height, image_width = image.shape[:2]

        if config.VERBOSE:
            print('\tFind textboxes...')
        
        # Running text detection
        # text images are in the following format: (image, box_center_point)
        t0_textbox = time.time()
        text_images, self.found_text_boxes = detector.find_text(image, config.craftnet, config.refinenet, config.GPU)
        td_textbox = time.time()

        self.runtime_findtext = (td_textbox-t0_textbox)*1000 # text detection runtime

        if config.VERBOSE:
            print(f'\tTextboxes found. ({self.runtime_findtext:.2f}ms)')

        # no text boxes found
        if len(text_images) == 0:
            return "?", "?"

        self.text_images = text_images

    def get_AP(self):
        if self.text_images is None or len(self.text_images) == 0:
            raise Exception("'text_images' is None. Text images was not initialized correctly.")

        self.AP = self.raw_AP = '?'

        # AP code is the first text image, most of the time 
        AP_candidate = self.text_images[0]
        AP_candidate_image = AP_candidate[0]

        if AP_candidate_image is None:
            return '?'

        # save the text box of the date candidate for debug purposes
        AP_text_box_top_left = (AP_candidate[1][0] - 1/2 * AP_candidate[0].shape[1], AP_candidate[1][1] - 1/2 * AP_candidate[0].shape[0])
        AP_text_box_bottom_right = (AP_candidate[1][0] + 1/2 * AP_candidate[0].shape[1], AP_candidate[1][1] + 1/2 * AP_candidate[0].shape[0])
        self.AP_text_box = AP_text_box_top_left, AP_text_box_bottom_right

        raw_AP = ocr.image_word_to_string(AP_candidate_image)
        self.raw_AP = raw_AP
        self.AP = self._post_process_AP(raw_AP)

        if config.DEBUG:
            cv2.namedWindow('ap candidate', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ap candidate', 1200,1000)
            cv2.imshow('ap candidate', AP_candidate_image)

        return self.AP

    def get_date(self):
        if self.text_images is None or len(self.text_images) == 0:
            raise Exception("'text_images' is None. Text images was not initialized correctly.")
    
        date_candidate = None
        self.date = self.raw_date = '?'

        # date will be the third or fourth or fifth or sixth element on a receipt
        # date is always on the left-side of the receipt
        # deciding between them by using these assumption
        if len(self.text_images) < 6:
            return '?'
        
        candidates = self.text_images[1], self.text_images[2], self.text_images[3], self.text_images[4], self.text_images[5]

        image_width = self.cropped_receipt.shape[1]

        # filter elements that are on the left side
        candidates_filtered = list(filter(lambda candidate: candidate[1][0] < image_width // 3, candidates))

        if len(candidates_filtered) == 0:
            candidates_filtered = filter(lambda candidate: candidate[1][0] < image_width // 2, candidates)

        # select the lowermost element from remaining candidates
        # reversing needed because the lowermost element has the highest 'y' value
        date_candidate = sorted(candidates_filtered, key=lambda candidate: candidate[1][1], reverse=True)[0]
        date_candidate_image = date_candidate[0]

        if date_candidate_image is None:
            return '?'

        # save the text box of the date candidate for debug purposes
        date_text_box_top_left = (date_candidate[1][0] - 1/2 * date_candidate[0].shape[1], date_candidate[1][1] - 1/2 * date_candidate[0].shape[0])
        date_text_box_bottom_right = (date_candidate[1][0] + 1/2 * date_candidate[0].shape[1], date_candidate[1][1] + 1/2 * date_candidate[0].shape[0])
        self.date_text_box = date_text_box_top_left, date_text_box_bottom_right

        raw_date = ocr.image_word_to_string(date_candidate_image)
        self.raw_date = raw_date
        self.date = self._post_process_date(raw_date)

        if config.DEBUG:
            cv2.namedWindow('date candidate', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('date candidate', 1200,1000)
            cv2.imshow('date candidate', date_candidate_image)

        return self.date

    def get_time(self):
        if self.text_images is None or len(self.text_images) == 0:
            raise Exception("'text_images' is None. Text images was not initialized correctly.")

        time_candidate = None
        self.time = self.raw_time = '?'

        # time will be the third or fourth or fifth or sixth element on a receipt
        # time is always on the right-side of the receipt
        # deciding between them by using these assumption
        if len(self.text_images) < 6:
            return '?'
        
        candidates = self.text_images[1], self.text_images[2], self.text_images[3], self.text_images[4], self.text_images[5]

        image_width = self.cropped_receipt.shape[1]

        # TODO delete
        # print(image_width, 2*(image_width//3))
        # for i, c in enumerate(candidates):
        #     print(f'test: {i}, x: {c[1][0]}')

        #     cv2.namedWindow(f'test: {i}', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow(f'test: {i}', 1200,1000)
        #     cv2.imshow(f'test: {i}', c[0])

        # filter elements that are on the right side
        candidates_filtered = list(filter(lambda candidate: candidate[1][0] > 2 * (image_width // 3), candidates))

        if len(candidates_filtered) == 0:
            candidates_filtered = filter(lambda candidate: candidate[1][0] > image_width // 2, candidates)

        # select the lowermost element from remaining candidates
        # list reversing needed because the lowermost element has the highest 'y' value
        time_candidate = sorted(candidates_filtered, key=lambda candidate: candidate[1][1], reverse=True)[0]
        time_candidate_image = time_candidate[0]

        if time_candidate_image is None:
            return '?'

        # save the text box of the time candidate for debug purposes
        time_text_box_top_left = (time_candidate[1][0] - 1/2 * time_candidate[0].shape[1], time_candidate[1][1] - 1/2 * time_candidate[0].shape[0])
        time_text_box_bottom_right = (time_candidate[1][0] + 1/2 * time_candidate[0].shape[1], time_candidate[1][1] + 1/2 * time_candidate[0].shape[0])
        self.time_text_box = time_text_box_top_left, time_text_box_bottom_right

        raw_time = ocr.image_word_to_string(time_candidate_image)
        self.raw_time = raw_time
        self.time = self._post_process_time(raw_time)

        if config.DEBUG:
            cv2.namedWindow('time candidate', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('time candidate', 1200,1000)
            cv2.imshow('time candidate', time_candidate_image)

        return self.time

    def draw_text_boxes(self, debug_image, offset=(0,0)):
        """Return with an opencv image, where we have drawn the found text boxes 
        onto the receipt."""
        if self.found_text_boxes is None:
            print("draw_text_boxes: No text found or estimation wasn't run till now, so returning with the parameter image")
            return debug_image

        for text_box in self.found_text_boxes:
            # get the top-left and bottom-right coords and shift them with the offset
            box_rectangle_coords = [(int(point[0])+offset[0], int(point[1])+offset[1]) for point in [np.min(text_box, axis=0), np.max(text_box, axis=0)]]
            debug_image = cv2.rectangle(debug_image, box_rectangle_coords[0], box_rectangle_coords[1], (0,255,255), 10)

        # draw the text boxes onto the debug_image
        shifted_AP_box = [(int(point[0])+offset[0], int(point[1])+offset[1]) for point in self.AP_text_box]
        shifted_date_box = [(int(point[0])+offset[0], int(point[1])+offset[1]) for point in self.date_text_box]
        shifted_time_box = [(int(point[0])+offset[0], int(point[1])+offset[1]) for point in self.time_text_box]
        debug_image = cv2.rectangle(debug_image, shifted_AP_box[0], shifted_AP_box[1], (0,255,0), 10)
        debug_image = cv2.rectangle(debug_image, shifted_date_box[0], shifted_date_box[1], (0,255,0), 10)
        debug_image = cv2.rectangle(debug_image, shifted_time_box[0], shifted_time_box[1], (0,255,0), 10)

        return debug_image