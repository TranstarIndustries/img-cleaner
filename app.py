import math
from typing import Tuple
from urllib.parse import quote

import requests

import cv2
import keras_ocr
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd


def midpoint(x1: int, y1: int, x2: int, y2:int) -> Tuple[int, int]:
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
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
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
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
                 
    return(inpainted_img)

def compute_grad(image_gray, mode:str)->np.ndarray:
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
        return (magmag/magmag.max()*255).astype(np.uint8)
        
    return (mag/mag.max()*255).astype(np.uint8)




def detect_draw(pipeline, image_gray, viz):
    img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    #read image from the an image path (a jpg/png file or an image url)
    # Prediction_groups is a list of (word, box) tuples
    b = pipeline.recognize([img])
    #print image with annotation and boxes
    if viz:
        keras_ocr.tools.drawAnnotations(image=img, predictions=b[0])
    
    
    x_min = int(min([b[0][i][1][:,0].min() for i in range(len(b[0]))]))
    x_max = int(max([b[0][i][1][:,0].max() for i in range(len(b[0]))]))

    y_min = int(min([b[0][i][1][:,1].min() for i in range(len(b[0]))]))
    y_max = int(max([b[0][i][1][:,1].max() for i in range(len(b[0]))]))
    return (x_min,y_min,x_max,y_max)

def remove(image_gray, bb, offset):
    # Create a mask of the same size as the image
    mask = np.ones_like(image_gray)*255

    # Draw a white rectangle on the mask within the bounding box
    cv2.rectangle(mask, (bb[0]-offset, bb[1]-offset), (bb[2]+offset, bb[3]+offset), (0,0,0), -1)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image_gray, mask)
    
    return masked_image

def sobel(pipeline, image_gray, offset):
    mag = compute_grad(image_gray,"single")
    bb = detect_draw(pipeline, mag, viz=True)
    masked_image = remove(image_gray, bb, offset)
    return masked_image

def img_cleaner_pipeline(img_path: str, sku: str, pipeline: keras_ocr.pipeline.Pipeline) -> np.ndarray:

    response = requests.get(img_path) 
    before_img = Image.open(BytesIO(response.content))
    image_gray = before_img.convert('L')
    # convert image to numpy array
    image_gray = np.array(image_gray)
    img_text_removed = inpaint_text(img_path, pipeline)
    #img_text_removed = sobel(pipeline, image_gray, offset=0)

    # save the image
    img_rgb = cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'text_removed/{sku}.jpg', img_rgb)

    #compare_fig(sku, before_img, img_text_removed)

def compare_fig(sku, before_img, img_text_removed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Displaying the first image
    axes[0].imshow(before_img)
    axes[0].set_title('Before')
    axes[0].axis('off')

    # Displaying the second image
    axes[1].imshow(img_text_removed)
    axes[1].set_title('After')
    axes[1].axis('off')

    plt.suptitle(f"SKU: {sku}")

    plt.savefig(f'compare_{sku}.jpg')


def get_image_path(sku: str) -> str:
    """
    This function takes in a SKU and returns the path to the image file.

    Parameters:
    sku (str): The SKU of the product

    Returns:
    image_path (str): The path to the image file
    """
    #image_path = f"images/{sku}.jpg"
    #apiToken = "nQev13tFnITKMn173PMEWmGQt4ogq1DdNYravcfRIXL7mqz3MHAZEPPJJYiRHIgeM/jkUnKwxC868FIUkyhOmw=="
    apiToken = "cDZdYXKGLNprECr6wDJ4cyJ2gm/pGwx4hziCx7JcmSe35D1xYlnoVKIzbvuKEXNTuTIvjPorti/6kIqoOZOjQg=="
    apiKey = "j495pu256SFZ7rdInW4X61vXLOelAbv9YZVtDaQh"
    headers = {"Authorization": f"Bearer {apiToken}", "x-api-key": apiKey}
    url = f"https://api.transend.us/product/?itemNumber={sku}"
    res = requests.get(url, headers=headers)
    response = res.json()
    image_path = response[0]['product']['images'][0]
    encoded_path = quote(image_path, safe='/:')
    return encoded_path



pipeline = keras_ocr.pipeline.Pipeline()
# load img-supression.csv and iterate through the rows
# if the Error Message contains "text, logo, graphic or watermark", then run the img_cleaner_pipeline function

# load img-supression.csv and iterate through the rows
df = pd.read_csv('img-supression.csv')
df.drop_duplicates(subset=['SKU'], keep=False, inplace=True)
sample = df.sample(10, replace=False)
for index, row in df.iterrows():
    print("cleaning: ", row['SKU'])
    sku = row['SKU']
    img_path = get_image_path(sku)
    if img_path == "https://cdn.transend.us/img/Keep%20Cool.png":
        continue
    if 'text, logo, graphic or watermark' in row['Error Details']:
        try:
          img_cleaner_pipeline(img_path, sku, pipeline)
          print(f"Image {img_path} cleaned for SKU {sku}")
        except Exception as e:
            print(f"Image not cleaned for SKU {row['SKU']}")
            print(e)
            continue
    else:
        print(f"Image not cleaned for SKU {row['SKU']}")
        print(f"Error Details : {row['Error Details']}")
        print(f"SKU: {row['SKU']}")