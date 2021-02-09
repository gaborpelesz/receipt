import os
import argparse
import time as chrono
import multiprocessing

import cv2
import numpy as np

import config
from receipt.extraction import ReceiptExtractor
from receipt.text.detector import load_text_detection_nets


def test(test_folder_path='/home/peleszgabor/Desktop/projects/blokkos/research_and_data/data/testing/test_set', 
        labels_path='/home/peleszgabor/Desktop/projects/blokkos/research_and_data/data/testing/test_labels.csv',
        regression_test_path=None,
        use_gpu=True,
        generate_regression_set=False):
    
    if not os.path.exists(test_folder_path):
        print(f"{test_folder_path}: Test folder does not exist! Returning...")
        return
    if not os.path.exists(labels_path):
        print(f'{labels_path}: Labels file does not exist! Returning...')
        return

    failed_cases = []
    runtimes = []
    regression_set = []
    regression_test_failed_cases = []

    if regression_test_path is not None:
        if not os.path.exists(regression_test_path):
            print(f'{regression_test_path}: Regression test file path was provided but the file does not exist! Returning...')
        if generate_regression_set:
            print('You are trying to generate and test against regression tests at the same time.')
            print('Testing against regression tests are higher order.')
            print('\tDisabling regression test generation...')
            generate_regression_set = False
        with open(regression_test_path) as f:
            regression_set = set(f.read().split('\n'))

    with open(labels_path, 'r') as f:
        # dict of '<file_name>': ['<ap>', '<date>', '<time>']
        labels = dict(map(lambda x: (x.split(',')[0], x.split(',')[1:]), f.read().split('\n')))
    
    extractor = ReceiptExtractor(use_gpu=use_gpu)
    load_text_detection_nets()

    image_names = os.listdir(test_folder_path)
    num_of_images = len(image_names)

    # TEST START
    for i, image_name in enumerate(image_names):
        print(f"{f'Testing: {i+1}/{num_of_images} -> {image_name}':40}", end='\r')

        if not os.path.exists(os.path.join(test_folder_path, image_name)):
            print('')
            print(f'File: {image_name} does not exist!')
            continue

        image = cv2.imread(os.path.join(test_folder_path, image_name), 1)

        if image is None:
            print('')
            print(f'Something happened while reading: "{image_name}". Maybe file format?')
            print('Continue...')
            continue

        try:
            receipt = extractor.extract_receipt(image)
            receipt.process()

            runtime_ocr_t0 = chrono.time()

            AP = receipt.get_AP()
            date = receipt.get_date()
            time = receipt.get_time()

            runtime_ocr = (chrono.time() - runtime_ocr_t0)*1000
        except Exception as e:
            print(e)
            raise Exception(f'Error occured while processing: {image_name}')

        if extractor.runtime_segment != 0 and receipt.runtime_findtext != 0 and runtime_ocr != 0:
            runtimes.append([extractor.runtime_segment, receipt.runtime_findtext, runtime_ocr])

        ground_truth = labels[image_name.split('.')[0]] # get rid of the file extension
        
        if AP != ground_truth[0] or date != ground_truth[1] or time != ground_truth[2]:
            failed_case = (image_name, (AP, ground_truth[0]), (date, ground_truth[1]), (time, ground_truth[2]))
            failed_cases.append(failed_case)
            if regression_test_path is not None and image_name in regression_set:
                regression_test_failed_cases.append(failed_case)
        elif generate_regression_set:
            regression_set.append(image_name)
    # TEST END

    num_of_failed_cases = len(failed_cases)
    print('') # new line

    ts = chrono.localtime() # timestamp struct
    test_result_file_name = f'{ts.tm_year}{ts.tm_mon:02}{ts.tm_mday:02}{ts.tm_hour:02}{ts.tm_min:02}{ts.tm_sec:02}'

    rt_extraction, rt_textsearch, rt_ocr = np.mean(runtimes, axis=0) # runtime results
    statistics_print = f"""Test results are written into folder: {test_result_file_name}
    Runtime statistics:
    \tExtraction    \tText search \tOCR
    \t({"GPU" if use_gpu else "CPU"}) {rt_extraction:.2f}ms    \t{rt_textsearch:.2f}ms \t{rt_ocr:.2f}ms
    Test result: {(1 - num_of_failed_cases/num_of_images) * 100:.1f}%"""

    if regression_test_path is not None:
        if len(regression_test_failed_cases) > 0:
            statistics_print += f"\nSome regression tests are failed. Success rate: {(1 - len(regression_test_failed_cases)/len(regression_set))*100:.1f}% \
                \nGet more information from the regression test result file.\n\n"
        else:
            statistics_print += "\nAll regression tests were successful.\n\n"
    else:
        statistics_print += "\n\n"

    os.mkdir(f'./test_results/{test_result_file_name}')

    with open(f'./test_results/{test_result_file_name}/test_result_{test_result_file_name}.txt', 'w+') as f:
        file_string = []
        for failed_case in failed_cases:
            row = f"""Failed case: {failed_case[0]}
            \t     \tresult    \tground truth
            \t  AP:\t{failed_case[1][0]}\t{failed_case[1][1]}
            \tdate:\t{failed_case[2][0]}\t{failed_case[2][1]}
            \ttime:\t{failed_case[3][0]}\t{failed_case[3][1]}"""
            file_string.append(row)
        f.write(statistics_print + '\n'.join(file_string))

    if len(regression_test_failed_cases) > 0:
        regression_test_header = "Regression test failed cases:\n\n"

        with open(f'./test_results/{test_result_file_name}/regression_test_result_{test_result_file_name}.txt', 'w+') as f:
            file_string = []
            for failed_case in regression_test_failed_cases:
                row = f"""Failed case: {failed_case[0]}
                \t     \tresult    \tground truth
                \t  AP:\t{failed_case[1][0]}\t{failed_case[1][1]}
                \tdate:\t{failed_case[2][0]}\t{failed_case[2][1]}
                \ttime:\t{failed_case[3][0]}\t{failed_case[3][1]}"""
                file_string.append(row)
            f.write(regression_test_header + '\n'.join(file_string))

    if generate_regression_set:
        print('Saving regression set...')
        with open('test_results/generated_regression_test_set.txt', 'w+') as f:
            f.write('\n'.join(regression_set))

    print(statistics_print[:-2]) # exclude last two new lines


def test_single(file_name, test_folder_path, use_gpu=False):
    # 936.jpg hard example working great
    # 924.jpg extreme segmentation
    # 684.jpg 391.jpg 818.jpg low-res detection

    config.DEBUG = True
    config.VERBOSE = True

    if not os.path.exists(os.path.join(test_folder_path, file_name)):
        print(f"{file_name}: File not exists!")
        print("Returning...")
        return

    extractor = ReceiptExtractor(use_gpu=use_gpu)
    load_text_detection_nets()

    print(f'Processing {file_name}')
    image = cv2.imread(os.path.join(test_folder_path, file_name), 1)
    debug_image = image.copy() # creating the debug image for API response

    print(f'image dimensions: {image.shape[1]}x{image.shape[0]}')

    t_start = chrono.time()
    receipt = extractor.extract_receipt(image)
    receipt.process()

    if config.VERBOSE:
        print('\tStart OCR...')

    ocr_start = chrono.time()

    AP = receipt.get_AP()
    date = receipt.get_date()
    time = receipt.get_time()

    # Drawing the receipt outline
    debug_image = extractor.draw_receipt_outline(debug_image)
    # Drawing the text boxes
    debug_image = receipt.draw_text_boxes(debug_image, extractor.rectangle_coords_of_receiptROI[0])

    cv2.namedWindow('receipt corners debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('receipt corners debug', 1000, 1600)
    cv2.imshow('receipt corners debug', debug_image)


    ocr_end = chrono.time()
    runtime_ocr = (ocr_end-ocr_start)*1000

    if config.VERBOSE:
        print('\tbefore postprocess')
        print(f'\t\tAP code: {receipt.raw_AP}')
        print(f'\t\t   date: {receipt.raw_date}')
        print(f'\t\t   time: {receipt.raw_time}')

        print('\tafter postprocess')
        print(f'\t\tAP code: {receipt.AP}')
        print(f'\t\t   date: {receipt.date}')
        print(f'\t\t   time: {receipt.time}')
        
        print(f'\tFinished OCR. ({runtime_ocr:.2f}ms)') # time

    runtime = chrono.time()-t_start
    print(f'Receipt OCR full runtime: {runtime*1000:.2f}ms')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Test recognition')

    parser.add_argument('-S', '--single', type=str, help='Test a specific image.')
    parser.add_argument('--test-path', type=str, default='/home/peleszgabor/Desktop/projects/blokkos/research_and_data/data/testing/test_set',
        help='The path to the folder, where the test files are stored')
    parser.add_argument('--labels-path', type=str, default='/home/peleszgabor/Desktop/projects/blokkos/research_and_data/data/testing/test_labels.csv',
        help='The path to the file, where the corresponding test labels are stored.')
    parser.add_argument('--regression-path', type=str, default='/home/peleszgabor/Desktop/projects/blokkos/research_and_data/data/testing/regression_test_set.txt',
        help='The path to the file, where the regression test image names are stored.')
    parser.add_argument('-R', '--test-regression', action='store_true', default=False, help='Wether to test against the regression test.')
    parser.add_argument('--gpu', action='store_true', default=False, help='Wether to use gpu for segmentation.')
    parser.add_argument('--gen-regression', action='store_true', default=False, 
        help='Generate regression test data from the successful recognitions')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.gpu:
        config.GPU = True

    if args.single is not None:
        test_single(file_name=args.single, test_folder_path=args.test_path, use_gpu=args.gpu)
    else:
        test(test_folder_path=args.test_path, labels_path=args.labels_path, use_gpu=args.gpu, regression_test_path=args.regression_path if args.test_regression else None, generate_regression_set=args.gen_regression)