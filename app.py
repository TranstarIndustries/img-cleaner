import os
from urllib.parse import quote

import keras_ocr
import matplotlib.pyplot as plt
import pandas as pd
import requests

from pipelines.img_resize import img_resize_pipeline
from pipelines.text_removal import text_removal_pipeline


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


def get_transend_image_uri(sku: str) -> str:
    """
    This function takes in a SKU and returns the path to the image file.

    Parameters:
    sku (str): The SKU of the product

    Returns:
    image_path (str): The path to the image file
    """
    api_token = os.environ.get("TRANSEND_API_TOKEN")
    api_key = os.environ.get("TRANSEND_API_KEY")
    headers = {"Authorization": f"Bearer {api_token}", "x-api-key": api_key}
    url = f"https://api.transend.us/product/?itemNumber={sku}"
    res = requests.get(url, headers=headers)
    response = res.json()
    image_path = response[0]['product']['images'][0]
    encoded_path = quote(image_path, safe='/:')
    return encoded_path



pipeline = keras_ocr.pipeline.Pipeline()
# load img-supression.csv and iterate through the rows
df = pd.read_csv('img-supression.csv')
df.drop_duplicates(subset=['SKU'], keep=False, inplace=True)
#sample = df.sample(10, replace=False)
for index, row in df.iterrows():
    print("cleaning: ", row['SKU'])
    sku = row['SKU']
    img_path = get_transend_image_uri(sku)
    if img_path == "https://cdn.transend.us/img/Keep%20Cool.png":
        continue
    if 'text, logo, graphic or watermark' in row['Error Details']:
        try:
          text_removal_pipeline(img_path, sku, pipeline)
          print(f"Image {img_path} cleaned for SKU {sku}")
        except Exception as e:
            print(f"Image not cleaned for SKU {row['SKU']}")
            print(e)
            continue
    if 'too small in the image frame' in row['Error Details']:
        try:
          img_resize_pipeline(img_path, sku, pipeline)
          print(f"Image {img_path} cleaned for SKU {sku}")
        except Exception as e:
            print(f"Image not re-sized for SKU {row['SKU']}")
            print(e)
            continue
    else:
        print(f"Image not cleaned for SKU {row['SKU']}")
        print(f"Error Details : {row['Error Details']}")
        print(f"SKU: {row['SKU']}")