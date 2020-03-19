import numpy as np
import cv2
import time
import pytesseract

class Receipt():
    def __init__(self, cropped_receipt):
        self.cropped_receipt = cropped_receipt

    def get_AP(self):
        pass

    def get_date(self):
        pass

    def get_all_text(self):
        img = self.cropped_receipt[3*self.cropped_receipt.shape[0]//4:, :]

        t_start = time.time()
        ocr_text = pytesseract.image_to_string(img)
        print(f'{(time.time()-t_start)*1000:.3f} ms')

        with open('test.txt', 'w') as f:
            f.write(ocr_text)

    def get_text(self, text_images):
        image = text_images

        t_start = time.time()
        config='--psm 8'
        ocr_text = pytesseract.image_to_string(image, config=config)
        print(f'ocr: {(time.time()-t_start)*1000:.3f} ms')

        with open('test.txt', 'w') as f:
            f.write(ocr_text)

    def find_text(self):
        img = self.cropped_receipt[3*self.cropped_receipt.shape[0]//4:, :]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)


        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


        # using RETR_EXTERNAL instead of RETR_CCOMP
        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #For opencv 3+ comment the previous line and uncomment the following line
        #_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mask = np.zeros(bw.shape, dtype=np.uint8)

        text_boxes = []
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y+h, x:x+w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

            if r > 0.45 and w > 16 and h > 16:
                rect = cv2.minAreaRect(contours[idx])

                text_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                convex_hull = cv2.convexHull(contours[idx])
                cv2.drawContours(text_mask, [convex_hull], -1, 255, -1)
                smoothed_mask = cv2.dilate(text_mask, kernel=np.ones((4,4),dtype=np.uint8), iterations=4)
                text_boxes.append((rect, smoothed_mask))
  

        text_images = []
        for text_box in text_boxes[:10]:
            angle = -text_box[0][2] if text_box[0][2] > -45 else 270-text_box[0][2]

            rotated_image, rotated_mask = rotate([img, text_box[1]], angle, center=text_box[0][0])

            contour = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            x, y, w, h = cv2.boundingRect(contour)

            rotated_mask = rotated_mask[y:y+h,x:x+w]
            rotated_mask = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)
            smoothed_image = cv2.bitwise_and(rotated_image[y:y+h,x:x+w], rotated_mask)
            smoothed_image = cv2.add(smoothed_image, cv2.bitwise_not(rotated_mask))
            
            text_images.append(smoothed_image)


        cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('rotated', (1500,1500))
        cv2.imshow('rotated', smoothed_image)

        for i, text_box in enumerate(text_boxes):
            box = cv2.boxPoints(text_box[0])
            box = np.int0(box)

            img = cv2.putText(img, f'{i+1}', tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.drawContours(img,[box],0,(0,0,255),2)

        cv2.namedWindow('grad', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('grad', 1200,1000)
        cv2.imshow('grad', grad)
        cv2.namedWindow('connected', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('connected', 1200,1000)
        cv2.imshow('connected', connected)
        cv2.namedWindow('show', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('show', (1500,1500))
        cv2.imshow('show', img)
        return text_images


def rotate(images, angle, center=None):
    if center is None:
        center = (images[0].shape[1]//2, images[0].shape[0]//2)

    (h, w) = images[0].shape[:2]
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # perform the actual rotation and return the images
    return [cv2.warpAffine(image, M, (nW, nH)) for image in images]