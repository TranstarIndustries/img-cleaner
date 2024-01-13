from io import BytesIO

import cv2
import keras_ocr
import numpy as np
import requests
from PIL import Image


def img_resize_pipeline(
    img_path: str, sku: str, pipeline: keras_ocr.pipeline.Pipeline
) -> np.ndarray:
    response = requests.get(img_path)
    before_img = np.array(Image.open(BytesIO(response.content)))
    img_zoom = zoom_at(before_img, zoom=1.5, angle=0, coord=None)
    # save the image
    img_rgb = cv2.cvtColor(img_zoom, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"img_resize/{sku}.jpg", img_rgb)


def zoom_at(img, zoom=1, angle=0, coord=None) -> np.ndarray:
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
