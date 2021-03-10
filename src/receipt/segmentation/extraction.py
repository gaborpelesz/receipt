import time
import numpy as np
import cv2
import tensorflow as tf

from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models.config import IMAGE_ORDERING

import config
from receipt.processing import Receipt
from receipt.utils.rotate import rotate

class ReceiptExtractor:
    def __init__(self, segmentation_model_path='models/segmentation/resnet50_unet_20200318', use_gpu=False):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        if not use_gpu:
            tf.config.experimental.set_visible_devices([], 'GPU') # forces operations on CPU

        self.model = model_from_checkpoint_path(segmentation_model_path)

        # warmup neural network, if not, the first prediction would be slow
        self._warmup()

        self.original_image = None
        self.receipt_corner_estimates = None
        self.rectangle_coords_of_receiptROI = None

        self.kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    def _warmup(self):
        warmup_image = cv2.imread('models/warmup/warmup.jpg', 1)
        _ = predict(model=self.model, image=warmup_image)

    def extract_receipt(self, image):
        """Finds and extracts a receipt, from an image.
        """
        self.original_image = image
        cropped_receipt = None

        if self.is_scanned(image):
            if config.VERBOSE:
                print("\tThe image looks like it was scanned.")
                print("\tText search based receipt extraction begins...")
            cropped_receipt = self._extract_by_text_search(image)
        else:
            t0_segment = time.time()
            if config.VERBOSE:
                print("\tReceipt segmentation with U-net begins...")

            cropped_receipt = self._extract_by_unet(image)

            td_segment = time.time()
            self.runtime_segment = (td_segment-t0_segment)*1000
            if config.VERBOSE:
                print(f"\tFinished segmentation. ({self.runtime_segment:.2f}ms)")

        if config.DEBUG:
            cv2.namedWindow('original', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('original', 1200, 1000)
            cv2.imshow('original', image)
            cv2.namedWindow('segmented', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('segmented', 1200, 1000)
            cv2.imshow('segmented', cropped_receipt)

        # TODO, we might just return the cropped receipt
        # and do the Receipt object instantiation outside
        if cropped_receipt is None:
            return None

        return Receipt(cropped_receipt) # create a new Receipt object

    def is_scanned(self, image):
        """Determines if an image is scanned or not.
        """

        # Convert image to gray and then resize it for performance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (400,400))

        # threshold for better white extraction (correct scanner machine errors and lightning)
        _, resized_image = cv2.threshold(resized_image, 220, 255, cv2.THRESH_BINARY)

        all_pixels = resized_image.shape[0] * resized_image.shape[1]

        # find and add up the white pixels
        white_pixels = np.sum(resized_image==255)

        # if white pixels are above 80% of the image
        # we consider it as a scanned image
        # (value is picked after testing)
        if white_pixels / all_pixels > 0.8:
            return True
        else:
            return False

    def _extract_by_unet(self, image):
        """Extraction based on the U-Net segmented image. This step
        aims to segment the hole receipt from the image. 
        
        Returns:
            - 'None' if no receipt were found during the extraction
            - else the rotated and extracted 'image' of the found receipt. 
              At this stage we assume that the image truly contains a receipt 
              or a receipt-like object that we couldn't distinguish from a real one.
        """
        segmentation_mask = self._unet_segment(image)

        # reduce noise, try to disconnect close artifacts from the receipt
        segmentation_mask = cv2.erode(segmentation_mask, self.kernel_5x5, iterations=5)
        segmentation_mask = cv2.dilate(segmentation_mask, self.kernel_5x5, iterations=5)

        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # No contours found
        if contours is None or len(contours) == 0:
            return None

        # Our best guess here is to operate with the biggest contour
        biggest_contour = None
        biggest_contour_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > biggest_contour_area:
                biggest_contour = contour
                biggest_contour_area = area

        contour_image_ratio = biggest_contour_area / (segmentation_mask.shape[0]*segmentation_mask.shape[1])

        # There aren't any receipts (artifacts) or the receipt is too small on the image.
        if contour_image_ratio < 0.1:
            return None

        # If the contour is above 90% of the image we think of it as the receipt
        # covers most of the image and it might be distorted in all sorts of ways
        # If it does not cover 90%, we need another step of confirmation which is
        # checking the number of corners in order to eliminate irregular shapes
        if contour_image_ratio < 0.9:
            num_corners_lower_bound  = len(self.estimate_corners(biggest_contour, 0.01))
            num_corners_upper_bound  = len(self.estimate_corners(biggest_contour, 0.04))

            if num_corners_upper_bound <= 3 or num_corners_upper_bound >= 7:
                return None
            if num_corners_lower_bound - num_corners_upper_bound >= 10:
                return None

        # 0.015 is a good corner estimation value
        # there are cases where we don't want the 4 point rectangular shape
        # for example cases where the receipt is bending and might be 6 points,
        # and we want to minimize the background so we are extraction along those 6.
        # Little artifacts can also cause unwanted corners, but we might can still
        # process the receipt extracting relevant text info
        self.receipt_corner_estimates = self.estimate_corners(biggest_contour, 0.015)

        # TODO, if we have 4 corners, DO 4 point transform. Else do the usual below.

        center, _, angle = cv2.minAreaRect(self.receipt_corner_estimates)

        smoothed_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(smoothed_mask, [cv2.convexHull(self.receipt_corner_estimates)], -1, 255, -1)

        rotated_image, rotated_mask = rotate([image, smoothed_mask], 270-angle, center)

        contour = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        x, y, w, h = cv2.boundingRect(contour)

        rotated_mask = rotated_mask[y:y+h,x:x+w]
        rotated_mask = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)
        smoothed_image = cv2.bitwise_and(rotated_image[y:y+h,x:x+w], rotated_mask)

        # If width is greater than height we rotate 90 clockwise
        # although it could have been a 90 anti-clockwise rotation...
        if smoothed_image.shape[1] > smoothed_image.shape[0]:
            smoothed_image = rotate([smoothed_image], 90)[0]

        return smoothed_image

    def _unet_segment(self, image_to_segment):
        original_height, original_width = image_to_segment.shape[0], image_to_segment.shape[1] 

        # Making the prediction over the image with the trained model
        pr = predict(model=self.model, image=image_to_segment)

        # Transform prediction
        seg_img = np.zeros((240, 176), dtype=np.uint8)
        seg_img[:, :] += ((pr == 1)*(255)).astype('uint8')

        # Upscale
        seg_img = cv2.resize(seg_img, (original_width, original_height))

        # after upscale the image won't be binary and gray shaded artifacts appear
        # to compensate thresholding at 254 is okay
        _, seg_img = cv2.threshold(seg_img, 254, 255, cv2.THRESH_BINARY)

        return seg_img

    def _extract_by_text_search(self, image):
        # TODO implement text extraction
        return None

    # TODO not sure perspective transform is needed yet
    def perspective_transform(self):
        """Extracts an object from an image by 4 corner points.
        """
        pass

    def estimate_corners(self, cnt, eps=0.01):
        epsilon = eps * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        return approx.reshape((approx.shape[0], approx.shape[2]))

    def draw_receipt_outline(self, debug_image):
        """Return with an opencv image, where we have drawn the estimated 
            receipt corners and the border of the receipt."""
        if self.receipt_corner_estimates is None:
            print("draw_receipt_corners: No receipt corner estimates were specified, so returning parameter image.")
            return self.debug_image

        # Drawing the outline of the receipt
        debug_image = cv2.polylines(debug_image, [self.receipt_corner_estimates], True, (0,0,255), 10, cv2.LINE_AA)

        # Drawing the area of interest, the 1/4 rectangle of the receipt
        rectangle_topleft = np.min(self.receipt_corner_estimates, axis=0)
        rectangle_bottomright = np.max(self.receipt_corner_estimates, axis=0)
        rectangle_topleft[1] += (3/4) * (rectangle_bottomright[1] - rectangle_topleft[1]) # moving the topleft coordinate down so the rectangle takes 1/4 of the receipt
        cv2.rectangle(debug_image, tuple(rectangle_topleft), tuple(rectangle_bottomright), (255,255,0), 10)

        self.rectangle_coords_of_receiptROI = [tuple(rectangle_topleft), tuple(rectangle_bottomright)]

        return debug_image

# ligthweight implementation of keras_segmentation predict function
def predict(model=None , image=None , out_fname=None):
    output_width = model.output_width
    output_height  = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array( image , input_width  , input_height , ordering=IMAGE_ORDERING )
    pr = model.predict( np.array([x]) )[0]

    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )

    return pr