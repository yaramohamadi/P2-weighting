import os
from PIL import Image

def resize_images(input_folder, output_folder, new_size=(256, 256)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add more image formats if needed
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Open the image
                with Image.open(input_path) as img:
                    # Check if the image is 512x512
                    if img.size == (512, 512):
                        # Resize the image
                        resized_img = img.resize(new_size, Image.LANCZOS)
                        # Save the resized image to the output folder
                        resized_img.save(output_path)
                        print(f"Resized and saved {filename} to {output_folder}")
                    else:
                        print(f"Skipping {filename} as it is not 512x512")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
input_folder = '/export/livia/home/vision/Ymohammadi/Dataset/cat10/'
output_folder = '/export/livia/home/vision/Ymohammadi/Dataset/cat102/'
resize_images(input_folder, output_folder)