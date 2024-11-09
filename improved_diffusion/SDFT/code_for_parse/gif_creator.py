from PIL import Image
import os
import math

def create_image_grid(input_folder, output_image_path, grid_size=(5, 10), image_size=(100, 100)):
    # List and sort all image files numerically by name assuming they are named 0 to 49
    image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))], 
                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # Ensure there are enough images for the grid
    if len(image_files) < grid_size[0] * grid_size[1]:
        raise ValueError("Not enough images to fill the grid.")

    # Create a blank canvas for the grid
    grid_width = grid_size[1] * image_size[0]
    grid_height = grid_size[0] * image_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Loop through and place each image on the grid
    for i, image_path in enumerate(image_files[:grid_size[0] * grid_size[1]]):
        img = Image.open(image_path).resize(image_size)
        x = (i % grid_size[1]) * image_size[0]
        y = (i // grid_size[1]) * image_size[1]
        grid_image.paste(img, (x, y))

    # Save the final grid image
    grid_image.save(output_image_path)
    print(f"Grid image saved as {output_image_path}")

# Set the folder path containing the images and the output path
input_folder = 'path/to/your/images'
output_image_path = 'path/to/save/grid_image.jpg'

# Create the image grid
create_image_grid(input_folder, output_image_path, grid_size=(5, 10), image_size=(100, 100))
