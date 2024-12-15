import os
from PIL import Image
import math

# Folder containing the images
image_folder = "path_to_your_image_folder"
output_image_path = "output_image_grid.png"

# Parameters for the grid
grid_columns = 10  # Number of columns in the grid
image_size = (32, 32)  # Resize each image to (width, height)

# Get a sorted list of image paths
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# Load and resize images
images = [Image.open(img).resize(image_size) for img in image_files]

# Calculate grid dimensions
grid_rows = math.ceil(len(images) / grid_columns)
grid_width = grid_columns * image_size[0]
grid_height = grid_rows * image_size[1]

# Create a blank canvas for the grid
grid_image = Image.new('RGB', (grid_width, grid_height))

# Paste images into the grid
for idx, img in enumerate(images):
    x_offset = (idx % grid_columns) * image_size[0]
    y_offset = (idx // grid_columns) * image_size[1]
    grid_image.paste(img, (x_offset, y_offset))

# Save the resulting grid image
grid_image.save(output_image_path)
print(f"Grid image saved to {output_image_path}")