import time
import json
from dotenv import load_dotenv
from flask import Flask, request, redirect, jsonify
from flask_api import FlaskAPI, status

app = FlaskAPI(__name__)
extractor = ReceiptExtractor()

@app.route('/process_receipt', methods=['POST'])
def hello_world():
    if request.method == 'POST' and request.is_json:
        return {
                'TODO': 'Hello world!'
            }, status.HTTP_500_INTERNAL_SERVER_ERROR
        
def process_receipt():
    receipt = extractor.extract_receipt(img)
    ap_code = receipt.get_AP()
    date = receipt.get_date()


if __name__ == '__main__':
    load_dotenv(override=True)
    app.run(host='0.0.0.0', port=3000)