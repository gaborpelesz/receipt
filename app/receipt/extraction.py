import time
import numpy as np
import cv2
import tensorflow as tf

from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.data_utils.data_loader import get_image_arr
from keras_segmentation.models.config import IMAGE_ORDERING

from receipt.processing import Receipt

class ReceiptExtractor:
    def __init__(self, segmentation_model_path='/home/peleszgabor/Desktop/projects/blokkos/alpha1.0/app/models/segmentation/resnet50_unet_20200318', use_gpu=False):
        # TODO warmup neural network, if not the first prediction will be slow
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        if not use_gpu:
            tf.config.experimental.set_visible_devices([], 'GPU') # forces operations on CPU

        self.model = model_from_checkpoint_path(segmentation_model_path)
        self._warmup()

    def _warmup(self):
        warmup_image = cv2.imread('app/models/warmup/warmup.jpg', 1)
        _ = predict(model=self.model, image=warmup_image)

    def is_scanned(self, image):
        """Determines if an image is scanned or not.
        """

        # Convert image to gray and then resize it for performance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (400,400))
        # threshold for better white extraction (correct scanner machine errors and lightning)
        _, resized_image = cv2.threshold(resized_image, 220, 255, cv2.THRESH_BINARY)

        all_pixels = resized_image.shape[0] * resized_image.shape[1]

        # create an array with True values where the images contained a 255
        # then test whether all array elements along a 
        # given axis (pixelwise in this case) evaluate to True
        white_pixels = np.sum(resized_image==255)

        # if white pixels are above 80% of the image
        # we consider it as a scanned image
        if white_pixels / all_pixels > 0.8:
            return True
        else:
            return False

    def extract_receipt(self, image):
        """Finds and extracts a receipt, from an image.
        """
        cropped_receipt = None

        if self.is_scanned(image):
            cropped_receipt = self.extract_by_text(image)
        else:
            cropped_receipt = self.extract_by_segment(image)

        return Receipt(cropped_receipt) # create a new Receipt object

    def extract_by_segment(self, image):
        """Extraction based on the segmented image. Segmentation
        aims to segment the hole receipt on the image. At this stage
        we assume that the image truly contains a receipt.
        """
        t_start = time.time()

        segmentation_mask = self.segment_image(image)

        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Our best guess here is to operate with the biggest contour later on
        biggest_contour = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

        corners = self.estimate_corners(biggest_contour)

        center, _, angle = cv2.minAreaRect(corners)

        smoothed_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(smoothed_mask, [cv2.convexHull(corners)], -1, 255, -1)

        rotated_image, rotated_mask = rotate([image, smoothed_mask], 270-angle, center)

        contour = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        x, y, w, h = cv2.boundingRect(contour)

        rotated_mask = rotated_mask[y:y+h,x:x+w]
        rotated_mask = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)
        smoothed_image = cv2.bitwise_and(rotated_image[y:y+h,x:x+w], rotated_mask)

        # If width is greater than height we rotate 90 clockwise
        if smoothed_image.shape[1] > smoothed_image.shape[0]:
            smoothed_image = rotate([smoothed_image], 90)[0]


        # print(f'Runtime: {(time.time()-t_start)*1000:.3f}ms')

        # window_size = (1500,1500)
        # cv2.namedWindow('show', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('show', window_size)
        # cv2.imshow('show', image)
        # cv2.namedWindow('smoothed_image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('smoothed_image', window_size)
        # cv2.imshow('smoothed_image', smoothed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return Receipt(smoothed_image)



    def segment_image(self, image_to_segment):
        original_height, original_width = image_to_segment.shape[0], image_to_segment.shape[1] 

        # Making the prediction over the image with the trained model
        pr = predict(model=self.model, image=image_to_segment)

        seg_img = np.zeros((240, 176), dtype=np.uint8)
        seg_img[:, :] += ((pr == 1)*(255)).astype('uint8')
        seg_img = cv2.resize(seg_img, (original_width, original_height))

        return seg_img

    def extract_by_text(self):
        pass

    # TODO not sure perspective transform is needed yet
    def perspective_transform(self):
        """Extracts an object from an image by 4 corner points.
        """
        pass

    def estimate_corners(self, cnt):
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        approx = np.array([a[0] for a in approx])

        return approx

def predict(model=None , image=None , out_fname=None):
    output_width = model.output_width
    output_height  = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr( image , input_width  , input_height , odering=IMAGE_ORDERING )
    pr = model.predict( np.array([x]) )[0]
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )

    return pr


def rotate(images, angle, center=None):
    if center is None:
        center = (images[0].shape[1]//2, images[0].shape[0]//2)

    (h, w) = images[0].shape[:2]
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    M_inv = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # perform the actual rotation and return the images
    return [cv2.warpAffine(image, M, (nW, nH)) for image in images]