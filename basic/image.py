

import os
import cv2

# Path to the directory containing input image files
input_directory = '/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Testing/fire'

# Create a new directory for saving reshaped images
output_directory = '/home/chathuranga_basnayaka/Desktop/my/semantic/wild/ageuav/datare'
os.makedirs(output_directory, exist_ok=True)

# List all image files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Define the new size you want for the images (width, height)
new_size = (256, 256)

for image_file in image_files:
    input_image_path = os.path.join(input_directory, image_file)
    image = cv2.imread(input_image_path)

    # Resize the image using OpenCV's resize function
    reshaped_image = cv2.resize(image, new_size)

    # Save the reshaped image to the output directory
    output_image_path = os.path.join(output_directory, f"reshaped_{image_file}")
    cv2.imwrite(output_image_path, reshaped_image)
