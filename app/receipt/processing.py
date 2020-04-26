import time
import re
import cv2

import config
from receipt import ocr

class Receipt():
    def __init__(self, cropped_receipt):
        self.date, self.AP = "?", "?"
        self.cropped_receipt = cropped_receipt

        self.runtime_ocr = 0
        self.runtime_findtext = 0

    def _post_process_AP(self, AP_code):
        if len(AP_code) < 8:
            return AP_code
        
        post_processed = AP_code
        post_processed = post_processed.replace('d', '0')
        post_processed = post_processed.replace('D', '0')
        post_processed = post_processed.replace('N', '0')
        post_processed = post_processed.replace('O', '0')
        post_processed = post_processed.replace('Q', '0')
        post_processed = post_processed.replace('V', '0')
        post_processed = post_processed.replace('I', '1')
        post_processed = post_processed.replace('L', '1')
        post_processed = post_processed.replace('T', '1')
        post_processed = post_processed.replace('?', '1')
        post_processed = post_processed.replace('B', '3')
        post_processed = post_processed.replace('A', '4')
        post_processed = post_processed.replace('S', '5')
        post_processed = post_processed.replace('§', '5')
        post_processed = post_processed.replace('b', '6')
        post_processed = post_processed.replace('G', '6')

        match = re.match('.+([0-9])[^0-9]*$', post_processed)

        if match is None:
            return post_processed

        last_digit_idx = match.start(1)

        post_processed = post_processed[last_digit_idx-7:last_digit_idx+1]

        return 'A' + post_processed

    def _post_process_date(self, date):
        post_processed = date

        post_processed = post_processed.replace('Q', '0')
        post_processed = post_processed.replace('A', '4')
        post_processed = post_processed.replace('G', '6')

        date_candidate = re.findall('([0-9]{0,2})', post_processed)

        if date_candidate is None:
            return post_processed

        date_candidate = [i for i in date_candidate if i] # remove empty strings

        if len(date_candidate) < 3:
            return post_processed

        date_candidate = date_candidate[-3:] 

        # if found then the date will be in form: [year, month, day] where 'year' has only the last two digits
        # assuming that this application will not be used in 2100<
        return f'20{date_candidate[0]}.{date_candidate[1]}.{date_candidate[2]}.'

    def get_date_and_AP(self):
        image = self.cropped_receipt[3*self.cropped_receipt.shape[0]//4:, :]
        image_height, image_width = image.shape[:2]

        if config.VERBOSE:
            print('\tFind textboxes...')
        t0_textbox = time.time()
        text_images = ocr.find_text(image)
        td_textbox = time.time()
        self.runtime_findtext = (td_textbox-t0_textbox)*1000
        if config.VERBOSE:
            print(f'\tTextboxes found. ({self.runtime_findtext:.2f}ms)')

        if len(text_images) == 0:
            return "?", "?"

        AP_candidate = text_images[0][0]
        date_candidate = None

        for text_image in text_images[1:]:
            if text_image[1][0] < image_width//3:
                date_candidate = text_image[0]
                break

        if config.VERBOSE:
            print('\tStart OCR...')
        ocr_start = time.time()
        raw_AP = ocr.image_word_to_string(AP_candidate)
        self.AP = self._post_process_AP(raw_AP)

        raw_date = ""

        if date_candidate is not None:
            raw_date = ocr.image_word_to_string(date_candidate)
            self.date = self._post_process_date(raw_date)
        ocr_end = time.time()
        self.runtime_ocr = (ocr_end-ocr_start)*1000

        if config.VERBOSE:
            print('\tbefore postprocess')
            print(f'\t\tAP code: {raw_AP}')
            print(f'\t\t   date: {raw_date}')

            print('\tafter postprocess')
            print(f'\t\tAP code: {self.AP}')
            print(f'\t\t   date: {self.date}')
            
            print(f'\tFinished OCR. ({self.runtime_ocr:.2f}ms)') # time
        
        return self.date, self.AP