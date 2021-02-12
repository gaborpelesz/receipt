import base64
import io
import sys
import cv2
from imageio import imread
from PIL import Image

def decode(b64_string, channels='RGB'):
    """Converting base64 string to cv2 image (image channel order: BGR).

        Args:
            b64_string: Representing the base64 encoded image.
        
        Return:
            A cv2 image, that we decoded from the base64 string,
            stored in a numpy array with image channel order: BGR.
    """

    decoded_string = base64.b64decode(b64_string)
    image = imread(io.BytesIO(decoded_string))

    if channels == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def encode(image, channels='RGB'):
    """Converting cv2 image to base64 string (image channel order: BGR).

        Args:
            image_bgr: Representing the image to be encoded. It needs to be in BGR
                channel order.
        
        Return:
            A base64 string, that we encoded from the image argument.
    """
    buffer = io.BytesIO()

    if channels == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    PIL_img = Image.fromarray(image)
    PIL_img.save(buffer, format="PNG")

    base64_img_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_img_string

##################
### DEPRECATED ###
### |||||||||| ###
### VVVVVVVVVV ###

def convert_base2image(b64_string):
    """Converting base64 string to cv2 image.

        Args:
            b64_string: Representing the base64 encoded image.
        
        Return:
            A cv2 image, that we decoded from the base64 string,
            stored in a numpy array.
    """
    print("WARNING! 'image_base64_converter.convert_base2image' function is deprecated.\
        Use 'image_base64_converter.decode_image' instead.", file=sys.stderr)

    # reconstruct image as an numpy array
    decoded_string = base64.b64decode(b64_string)
    img = imread(io.BytesIO(decoded_string))

    # finally convert RGB image to BGR for opencv
    # and save result
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2_img

def convert_image2base(img_bgr):
    print("WARNING! 'image_base64_converter.convert_image2base' function is deprecated.\
        Use 'image_base64_converter.encode_image' instead.", file=sys.stderr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    base64_img_string = base64.b64encode(buff.getvalue()).decode("utf-8")

    # FOR MORE EASILY DEBUG BEST_GUESS_IMAGE IN DEV MODE
    # with open("/app/test.txt", "w") as text_file:
    #     text_file.write(base64_img_string)

    return base64_img_string