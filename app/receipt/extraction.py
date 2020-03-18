import numpy as np
import cv2
from keras_segmentation.predict import model_from_checkpoint_path, predict

from app.receipt.processing import Receipt

class ReceiptExtractor:
    def __init__(self, segmentation_model_path='app/models/resnet50_unet_20200318'):
        self.model = model_from_checkpoint_path(segmentation_model_path)
        pass

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
        segmentation_mask = segment_image(image)

        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

    def segment_image(self, image_to_segment):
        original_height, original_width = image_to_segment.shape[0], image_to_segment.shape[1] 

        # Making the prediction over the image with the trained model
        pr = predict(
            model=model, 
            inp=image_to_segment
            )

        # Transforming the prediction mask so that it can be blended with
        #   the original image
        seg_img = np.zeros((240, 176, 3), dtype=np.uint8)
        colors = [
            (0,0,0),
            (255, 255, 255),
        ]

        for c in range(2):
            seg_img[:, :, 0] += ((pr[:, :] == c)*(colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c)*(colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c)*(colors[c][2])).astype('uint8')

        seg_img = cv2.resize(seg_img, (original_width, original_height))

        return seg_img

    def extract_by_text(self):
        pass

    def estimate_corners(self):
        pass

    # TODO not sure perspective transform is needed yet
    def perspective_transform(self):
        """Extracts an object from an image by 4 corner points.
        """
        pass