import os
import numpy as np
from PIL import Image
import argparse

def load_images_from_folder(folder_path, img_size=(64, 64)):
    """
    Loads images from a folder, resizes them, and converts them to a numpy array.

    :param folder_path: Path to the folder containing images.
    :param img_size: Desired size to resize the images (width, height).
    :return: A numpy array of images.
    """
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            with Image.open(img_path) as img:
                img = img.resize(img_size)  # Resize image
                img_array = np.array(img)  # Convert image to numpy array
                images.append(img_array)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images)

def create_npz_from_images(folder_path, output_file, img_size=(224, 224)):
    """
    Creates an NPZ file from a folder of images.

    :param folder_path: Path to the folder containing images.
    :param output_file: Output path for the NPZ file.
    :param img_size: Desired size to resize the images (width, height).
    """
    images = load_images_from_folder(folder_path, img_size)
    np.savez_compressed(output_file, images=images)
    print(f"NPZ file saved at {output_file}")

if __name__ == "__main__":

    create_npz_from_images(folder_path='/home/ymbahram/scratch/pokemon/pokemon10/', output_file='/home/ymbahram/scratch/pokemon/pokemon_10.npz', img_size=(64,64))