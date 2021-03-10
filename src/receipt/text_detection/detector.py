import cv2
import numpy as np
import pytesseract

import config
from receipt.utils.rotate import rotate

import craft_text_detector as craft

def load_text_detection_nets():
    config.craftnet  = craft.load_craftnet_model(config.GPU)
    config.refinenet = craft.load_refinenet_model(config.GPU)

def find_text(image, craftnet, refinenet, gpu: bool = False):
    text_images = []
    found_text_boxes = []

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[0] == 2:
        image = image[0]
    if image.shape[2] == 4:
        image = image[:, :, :3]

    prediction_result = craft.get_prediction(
        image=image,
        craft_net=craftnet,
        refine_net=refinenet,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=gpu,
        long_size=1280/2,
        poly=False
    )

    # No textbox found
    if prediction_result["boxes"] is None or len(prediction_result["boxes"]) == 0:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for box in prediction_result["boxes"]:
        box = box.astype(np.int32)

        box[0][0] = box[0][0] - image.shape[1]*0.003 # top left x
        box[0][1] = box[0][1] - image.shape[0]*0.0025 # top left y

        box[1][0] = box[1][0] + image.shape[1]*0.003 # top right x
        box[1][1] = box[1][1] - image.shape[0]*0.0025 # top right y

        box[2][0] = box[2][0] + image.shape[1]*0.003 # bottom right x
        box[2][1] = box[2][1] + image.shape[0]*0.0025 # bottom right y

        box[3][0] = box[3][0] - image.shape[1]*0.003 # bottom left x
        box[3][1] = box[3][1] + image.shape[0]*0.0025 # bottom left y

        cropped_image = crop_box(image, box)

        position = (box[0] + box[2]) // 2 # center of the textbox on the image

        text_images.append((cropped_image, position))
        found_text_boxes.append(box)

    if len(text_images) > 0:
        text_images.reverse() # interesting fact that if len(text_images) == 0 so text_images = [], then text_images.reverse() is None!

    if config.DEBUG:
        for i, box in enumerate(prediction_result["boxes"]):
            box = np.int0(box)

            image = cv2.putText(image, f'{i+1}', tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.drawContours(image,[box],0,(0,0,255),2)

        for i, text_image in enumerate(text_images[:10]):
            cv2.namedWindow(f'text image {i+1}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'text image {i+1}', 1200,1000)
            cv2.imshow(f'text image {i+1}', text_image[0])

        cv2.namedWindow('text boxes', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('text boxes', 1200,1000)
        cv2.imshow('text boxes', image)

    return text_images, found_text_boxes

def crop_box(image, box):
    # get width and height of the detected rectangle
    width = int(np.linalg.norm(box[0]-box[1]))
    height = int(np.linalg.norm(box[1]-box[2]))

    # box originally contains points as follows: top-left, top-right, bottom-right, bottom-left
    # perspective transform receives the points as follows: bottom left, top left, top right, bottom-right
    src_pts = np.array([box[3], box[0], box[1], box[2]], dtype="float32")

    dst_pts = np.array([[      0, height-1],
                        [      0,        0],
                        [width-1,        0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped