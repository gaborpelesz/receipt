import cv2
import numpy as np
import pytesseract

import config
from utils.rotate import rotate

import craft_text_detector as craft

def load_text_detection_nets():
    config.craftnet  = craft.load_craftnet_model(config.GPU)
    config.refinenet = craft.load_refinenet_model(config.GPU)

def find_text(image, craftnet, refinenet, gpu: bool = False):
    text_images = []

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

    text_images.reverse()

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

    return text_images

def crop_box(image, box):
    # get width and height of the detected rectangle
    width = int(np.linalg.norm(box[0]-box[1]))
    height = int(np.linalg.norm(box[1]-box[2]))

    # box originally contains points as follows: top-left, top-right, bottom-right, bottom-left
    # perspective transform receives the points as follows: bottom left, top left, top right, bottom-right
    src_pts = np.array([box[3], box[0], box[1], box[2]], dtype="float32")

    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

"""
def find_text(image, _, _1, _2):
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
        
        if r > 0.4 and w/im_width > 0.08 and h/im_height > 0.02:
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
        
        position = (text_box[0][0][0], text_box[0][0][1]) # center of the textbox on the image

        text_images.append((cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY), position))

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
        cv2.imshow('text boxes', image)

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
"""