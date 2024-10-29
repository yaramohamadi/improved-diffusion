import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_grid_image(folder_path, subfolder_choice, output_folder, grid_size=(5, 10), image_size=(64, 64)):
    """
    Create a grid image from a folder of images and save it.
    
    :param folder_path: Path to the folder containing subfolders with images.
    :param subfolder_choice: Choice of subfolder to select (_0, _0_1, or _0_05).
    :param output_folder: Output folder to save the grid images.
    :param grid_size: Number of rows and columns for the grid.
    :param image_size: Resize each image to this size.
    """
    folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
    
    for folder in folders:
        subfolder_path = os.path.join(folder_path, folder, subfolder_choice)
        image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.jpg') or f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

        # Create a blank canvas for the grid
        grid_image = Image.new('RGB', (grid_size[1] * image_size[0], grid_size[0] * image_size[1]))

        for i, image_file in enumerate(image_files):
            img = Image.open(os.path.join(subfolder_path, image_file)).resize(image_size)
            row, col = divmod(i, grid_size[1])
            grid_image.paste(img, (col * image_size[0], row * image_size[1]))

        # Add folder name as a title
        plt.figure(figsize=(10, 8))
        plt.imshow(grid_image)
        plt.title(f"Folder: {folder}", fontsize=16)
        plt.axis('off')
        
        # Save the output image
        output_image_path = os.path.join(output_folder, f'grid{subfolder_choice}_{folder}.png')
        plt.savefig(output_image_path)
        plt.close()

# Example usage:
folder_path = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_and_sampling/0_1/samples'  # Path to the main directory
subfolder_choice = '_0'  # Choose between '_0', '_0_1', or '_0_05'
output_folder = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_and_sampling/0_1'
create_grid_image(folder_path, subfolder_choice, output_folder)