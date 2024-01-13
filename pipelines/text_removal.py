import math
from io import BytesIO
from typing import Tuple

import cv2
import keras_ocr
import numpy as np
import requests
from PIL import Image


def text_removal_pipeline(
    img_path: str, sku: str, pipeline: keras_ocr.pipeline.Pipeline, dir="text_removed"
) -> np.ndarray:
    if "http" in img_path:
        response = requests.get(img_path)
        before_img = Image.open(BytesIO(response.content))
    else:
        before_img = Image.open(img_path)
    image_gray = before_img.convert("L")
    # convert image to numpy array
    image_gray = np.array(image_gray)
    img_text_removed = inpaint_text(img_path, pipeline)
    # img_text_removed = sobel(pipeline, image_gray, offset=0)

    # save the image
    img_rgb = cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{dir}/{sku}.jpg", img_rgb)

    # compare_fig(sku, before_img, img_text_removed)


def midpoint(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """
    This function takes in the coordinates of two points and returns the midpoint.

    Parameters:
    x1 (int): The x coordinate of the first point
    y1 (int): The y coordinate of the first point
    x2 (int): The x coordinate of the second point
    y2 (int): The y coordinate of the second point

    Returns:
    x_mid (int): The x coordinate of the midpoint
    y_mid (int): The y coordinate of the midpoint
    """
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def inpaint_text(img_path: str, pipeline: keras_ocr.pipeline.Pipeline) -> np.ndarray:
    """
    This function takes in an image path and a keras-ocr pipeline object and returns an inpainted image
    with the text removed.

    Parameters:
    img_path (str): The path to the image file
    pipeline (Pipeline): The keras-ocr pipeline object

    Returns:
    inpainted_img (numpy array): The inpainted image with the text removed
    """
    # read the image
    img = keras_ocr.tools.read(img_path)

    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([img])

    # Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        # For the line thickness, we will calculate the length of the line between
        # the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        # Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        inpainted_img = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    return inpainted_img


def compute_grad(image_gray, mode: str) -> np.ndarray:
    """
    Function to compute the gradients magnitude of the image
    set mode to "double" if you want to apply it two times
    """
    # Compute the gradients using the Sobel operator
    dx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude and direction of the gradients
    mag = np.sqrt(dx**2 + dy**2)

    if mode == "double":
        # Compute the gradients using the Sobel operator
        dx = cv2.Sobel(mag, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(mag, cv2.CV_64F, 0, 1, ksize=3)

        # Compute the magnitude and direction of the gradients
        magmag = np.sqrt(dx**2 + dy**2)
        return (magmag / magmag.max() * 255).astype(np.uint8)

    return (mag / mag.max() * 255).astype(np.uint8)


def detect_draw(pipeline, image_gray, viz):
    img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    # read image from the an image path (a jpg/png file or an image url)
    # Prediction_groups is a list of (word, box) tuples
    b = pipeline.recognize([img])
    # print image with annotation and boxes
    if viz:
        keras_ocr.tools.drawAnnotations(image=img, predictions=b[0])

    x_min = int(min([b[0][i][1][:, 0].min() for i in range(len(b[0]))]))
    x_max = int(max([b[0][i][1][:, 0].max() for i in range(len(b[0]))]))

    y_min = int(min([b[0][i][1][:, 1].min() for i in range(len(b[0]))]))
    y_max = int(max([b[0][i][1][:, 1].max() for i in range(len(b[0]))]))
    return (x_min, y_min, x_max, y_max)


def remove(image_gray, bb, offset):
    # Create a mask of the same size as the image
    mask = np.ones_like(image_gray) * 255

    # Draw a white rectangle on the mask within the bounding box
    cv2.rectangle(
        mask,
        (bb[0] - offset, bb[1] - offset),
        (bb[2] + offset, bb[3] + offset),
        (0, 0, 0),
        -1,
    )

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image_gray, mask)

    return masked_image


def sobel(pipeline, image_gray, offset):
    mag = compute_grad(image_gray, "single")
    bb = detect_draw(pipeline, mag, viz=True)
    masked_image = remove(image_gray, bb, offset)
    return masked_image
