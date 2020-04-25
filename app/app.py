import time
import io
from flask import Flask, request, jsonify
from flask_api import FlaskAPI, status
import cv2
import numpy as np

from receipt.extraction import ReceiptExtractor
import config
from utils import base64_image_converter

app = FlaskAPI(__name__)
extractor = ReceiptExtractor(use_gpu=config.GPU)

@app.route('/process_receipt', methods=['POST'])     
def process_receipt():
    if request.method == 'POST' and request.files.get('image'):
        start_time = time.time() # track runtime

        print('Decoding bytes to image...')
        t0_decode = time.time()
        image_bytes = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        print(f'Decoding completed. ({(time.time()-t0_decode)*1000:.2f}ms)')



        print('Extracting receipt from image...')
        t0_extraction = time.time()
        receipt = extractor.extract_receipt(image)
        print(f'Extraction completed. ({(time.time()-t0_extraction)*1000:.2f}ms)')

        if receipt is None:
            return jsonify(
                status='Success',
                receipt={
                    'date': '?',
                    'AP_code': '?'
                },
                runtime=f'{(time.time()-start_time)*1000:.2f}ms'
            ), status.HTTP_404_NOT_FOUND

        print('Start OCR on receipt...')
        t0_ocr = time.time()
        date, AP = receipt.get_date_and_AP()
        print(f'OCR completed. ({(time.time()-t0_ocr)*1000:.2f}ms)')

        return jsonify(
            status='Success',
            receipt={
                'date': date,
                'AP_code': AP,
            },
            runtime=f'{(time.time()-start_time)*1000:.2f}ms'
        ), status.HTTP_200_OK

    return jsonify(
        status='Failed',
        status_message='The request method was not "POST".' if request.method != 'POST' else 'The request is not a JSON request.'
    ), status.HTTP_400_BAD_REQUEST


if __name__ == '__main__':
    # threaded=false because if not the Flask would lock threads before tensorflow
    app.run(host='0.0.0.0', port=3000, threaded=False)