import os
import numpy as np
from PIL import Image

def images_to_npz(image_folder, output_file):
    image_list = []
    image_names = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add more image formats if needed
            image_path = os.path.join(image_folder, filename)
            try:
                with Image.open(image_path) as img:
                    img_array = np.array(img)
                    image_list.append(img_array)
                    image_names.append(filename)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    
    # Save the list of images and their corresponding filenames to an npz file
    np.savez(output_file, images=image_list, filenames=image_names)
    print(f"Saved {len(image_list)} images to {output_file}")

# Example usage:
image_folder = '/export/livia/home/vision/Ymohammadi/Dataset/sketch_cropped/'
output_file = '/export/livia/home/vision/Ymohammadi/Dataset/sketch.npz'
images_to_npz(image_folder, output_file)