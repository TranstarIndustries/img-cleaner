import pandas as pd
import zipfile
from pathlib import Path
from PIL import Image
from io import BytesIO
import json
import boto3
import base64
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

bedrock_runtime_client = boto3.client('bedrock-runtime', region_name='us-east-1' )

def create_vehicle_image(prompt, style_preset=None):
        try:
            body = json.dumps(
                {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text":prompt,   # Required
            #           "negativeText": "<text>"  # Optional
                    },
                    "imageGenerationConfig": {
                        "numberOfImages": 1,   # Range: 1 to 5 
                        "quality": "premium",  # Options: standard or premium
                        "height": 1024,         # Supported height list in the docs 
                        "width": 1024,         # Supported width list in the docs
                        "cfgScale": 7.5,       # Range: 1.0 (exclusive) to 10.0
                        "seed": 42             # Range: 0 to 214783647
                    }
                }
            )
            response = bedrock_runtime_client.invoke_model(
                body=body, 
                modelId="amazon.titan-image-generator-v1",
                accept="application/json", 
                contentType="application/json"
            )

            response_body = json.loads(response["body"].read())
            base64_image_data = response_body["images"][0]

            return base64_image_data

        except ClientError:
            logger.error("Couldn't invoke Titan Image Generator Model")
            raise
if __name__ == "__main__":
    # Load the spreadsheet data
    df = pd.read_excel('Vehicles_2005.xlsx')

    # Filter rows where BaseVehicleID is within a specified range
    filtered_df = df[(df['BaseVehicleID'] >= 18276) & (df['BaseVehicleID'] <= 18281)]

    # Create a zip file
    with zipfile.ZipFile('vehicles_images.zip', 'w') as zipf:
        for index, row in filtered_df.iterrows():
            # Assuming a function create_vehicle_image exists that takes year, make, model as input
            # and returns an image file named with BaseVehicleID
            prompt = f"{row['YearID']} {row['MakeName']} {row['ModelName']}, neutral background"
            b64_image = create_vehicle_image(prompt)
            
            image = Image.open(BytesIO(base64.b64decode(b64_image)))

            # Define the file name
            image_file_name = f"{row['BaseVehicleID']}.png"

            # Save the image file with the name as BaseVehicleID
            image_path = Path(image_file_name)
            image.save(image_path)

            # Add the image file to the zip
            zipf.write(image_path, arcname=image_path.name)

            # Optionally remove the image file after adding it to the zip if you don't want it to remain on disk
            image_path.unlink()
