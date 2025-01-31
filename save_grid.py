import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def create_image_grid(base_dir, output_dir):
    """
    Create a grid image for each g{num} folder where each row corresponds
    to the 10 images from subfolders inside the samples directory.

    Parameters:
    - base_dir: Path to the directory containing g{num} folders.
    - output_dir: Path to save the generated grid images.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all g{num} folders
    g_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('g')]

    for g_folder in g_folders:
        g_path = os.path.join(base_dir, g_folder, 'samples')
        if not os.path.exists(g_path):
            print(f"Skipping {g_folder}, no 'samples' folder found.")
            continue

        # List all numbered subfolders (e.g., 0, 25, 50, ..., 300)
        subfolders = sorted([f for f in os.listdir(g_path) if os.path.isdir(os.path.join(g_path, f))], key=lambda x: int(x))

        # Initialize a list to hold rows of the grid
        grid_rows = []

        for subfolder in subfolders:
            subfolder_path = os.path.join(g_path, subfolder)
            # Collect the 10 images (0 to 9)
            images = []
            for i in range(10):
                image_path = os.path.join(subfolder_path, f"{i}.png")
                if os.path.exists(image_path):
                    images.append(mpimg.imread(image_path))
                else:
                    print(f"Missing image: {image_path}, skipping.")
                    continue

            if len(images) == 10:
                grid_rows.append(images)
            else:
                print(f"Skipping subfolder {subfolder} in {g_folder} due to insufficient images.")
        
        if grid_rows:
            # Create the grid plot
            fig, axes = plt.subplots(len(grid_rows), 10, figsize=(20, len(grid_rows) * 2))

            for row_idx, row_images in enumerate(grid_rows):
                for col_idx, img in enumerate(row_images):
                    ax = axes[row_idx, col_idx]
                    ax.imshow(img)
                    ax.axis("off")

            # Adjust spacing and save the grid image
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{g_folder}_grid.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved grid image for {g_folder} to {output_path}")
        else:
            print(f"No valid rows found for {g_folder}, skipping.")

# Define paths
base_directory = "/home/ens/AT74470/clf_results/clf_xs_xt/sigmoid/"  # Replace with your base directory path
output_directory = "/home/ens/AT74470/clf_results/clf_xs_xt/sigmoid/"  # Replace with your output directory path

# Generate grids
create_image_grid(base_directory, output_directory)