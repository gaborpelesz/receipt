import time
import io
from flask import Flask, request, jsonify
from flask_api import FlaskAPI, status
import cv2
import numpy as np

from receipt.extraction import ReceiptExtractor
from receipt.text.detector import load_text_detection_nets
import config
from utils import base64_image_converter

app = FlaskAPI(__name__)
load_text_detection_nets()
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
                status='Failed',
                status_message='No receipt found on the image.',
                receipt={
                    'AP': '?',
                    'date': '?',
                    'time': '?'
                },
                runtime=f'{(time.time()-start_time)*1000:.2f}ms'
            ), status.HTTP_404_NOT_FOUND

        print('Processing the receipt...')
        t0_processing = time.time()
        receipt.process()
        print(f'Processing completed. ({(time.time()-t0_processing)*1000:.2f}ms)')

        print('Extraction of predefined fields...')
        receipt_AP = receipt.get_AP()
        receipt_date = receipt.get_date()
        receipt_time = receipt.get_time()
        print('Extraction of AP, date, time completed.')

        return jsonify(
            status='Success',
            receipt={
                    'AP': receipt_AP,
                    'date': receipt_date,
                    'time': receipt_time
            },
            runtime=f'{(time.time()-start_time)*1000:.2f}ms'
        ), status.HTTP_200_OK

    return jsonify(
        status='Failed',
        status_message='The request method was not "POST".' if request.method != 'POST' else 'The request is not a JSON request.'
    ), status.HTTP_400_BAD_REQUEST


if __name__ == '__main__':
    # threaded=false because if not, Flask would lock threads before tensorflow
    app.run(host='0.0.0.0', port=3000, threaded=False)