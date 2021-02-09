import time
import io
import sys
from flask import Flask, request, jsonify, make_response
from flask_api import FlaskAPI, status
import cv2
import numpy as np
import base64

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
        request_json_content = request.get_json()

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
                status_message='Receipt not found on the image.',
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
        try:
            receipt_AP = receipt.get_AP()
            receipt_date = receipt.get_date()
            receipt_time = receipt.get_time()
        except Exception as e:
            print(e, file=sys.stderr)
            return jsonify(
                status='Success',
                status_message='Receipt not found on the image.',
                receipt={
                        'AP': '?',
                        'date': '?',
                        'time': '?'
                },
                runtime=f'{(time.time()-start_time)*1000:.2f}ms'
            ), status.HTTP_404_NOT_FOUND
        print('Extraction of AP, date, time completed.')

        # if API asked, create debug image
        try:
            if request.args.get('debug') == "True":
                print("Drawing function ran...", file=sys.stderr)
                # Drawing the receipt outline
                debug_image = image.copy()
                debug_image = extractor.draw_receipt_outline(debug_image)
                # Drawing the text boxes
                debug_image = receipt.draw_text_boxes(debug_image, extractor.rectangle_coords_of_receiptROI[0])

                debug_image = cv2.resize(debug_image, (1000, int(1000/debug_image.shape[1]*debug_image.shape[0])))

                debug_image_binary = cv2.imencode('.png', debug_image)[1]
                debug_image_base64 = base64.b64encode(debug_image_binary).decode()
                
                return jsonify(
                    status='Success',
                    status_message='Successful processing of the receipt on the image.',
                    receipt={
                            'AP': receipt_AP,
                            'date': receipt_date,
                            'time': receipt_time
                    },
                    runtime=f'{(time.time()-start_time)*1000:.2f}ms',
                    debug_image=debug_image_base64
                ), status.HTTP_200_OK

        except Exception as e:
            print(e, file=sys.stderr)
            print("Tried but exception occured.", file=sys.stderr)


        return jsonify(
            status='Success',
            status_message='Successful processing of the receipt on the image.',
            receipt={
                    'AP': receipt_AP,
                    'date': receipt_date,
                    'time': receipt_time
            },
            runtime=f'{(time.time()-start_time)*1000:.2f}ms'
        ), status.HTTP_200_OK

    if request.method != 'POST':
        return jsonify(
            status='Failed',
            status_message='The request method was not "POST".'
        ), status.HTTP_400_BAD_REQUEST
    elif request.files.get('image'):
        return jsonify(
            status='Failed',
            status_message='The request method was not "POST".'
        ), status.HTTP_400_BAD_REQUEST

    return jsonify(
        status='Failed',
        status_message='Unknown internal server error. Please save the configuration and contact gaborpelesz@gmail.com'
    ), status.HTTP_500_INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    # threaded=false because if not, Flask would lock threads before tensorflow
    app.run(host='0.0.0.0', port=3000, threaded=False)