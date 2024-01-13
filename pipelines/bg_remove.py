import io

from PIL import Image
from rembg import remove


def remove_bg(input_path: str, output_path: str):
    """
    Removes the background from an image and saves the result to a new image file.
    """
    with open(input_path, "rb") as input_file:
        input_image = input_file.read()
        output_image = remove(input_image) 
        # Create an image object from the output image
        processed_image = Image.open(io.BytesIO(output_image))
        # Check if the image has an alpha channel
        if processed_image.mode == 'RGBA':
            # Create a white background image with the same size as the processed image
            white_bg = Image.new("RGB", processed_image.size, "WHITE")

            # Blend the processed image with the white background
            white_bg.paste(processed_image, (0, 0), processed_image)
            final_image = white_bg
        else:
            final_image = processed_image
        # Save the final image
        final_image.save(output_path, format="JPEG")
    

if __name__ == "__main__":
    import os
    import sys

    #input_path = sys.argv[1]
    def list_files(directory):
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    directory_path = 'keeper'  # Replace with your directory path
    files = list_files(directory_path)
    error_count = 0
    for file in files:
        print("Processing file: " + file)
        input_path = directory_path + '/' + file
        output_path = input_path.replace(".jpg", "_no_bg.jpg")
        try:
            remove_bg(input_path, output_path)
        except Exception as e:
            print(e)
            print("Error processing file: " + input_path)
            error_count+=1
            continue
    print("Done!")
    print("Error count: " + str(error_count))