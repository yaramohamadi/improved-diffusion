import os
import shutil
import random
from pathlib import Path

def organize_images(data_dir_1, data_dir_2, val_data_dir_2, output_dir, val_size=1000):
    """
    Organize images from two domains into a new folder with labels based on domain
    and perform a train-validation split.
    
    :param data_dir_1: Path to the first domain's images.
    :param data_dir_2: Path to the second domain's training images.
    :param val_data_dir_2: Path to the second domain's validation images.
    :param output_dir: Path to the new folder for organized dataset.
    :param val_size: Number of samples to take for validation from each domain.
    """
    # Set up paths for the new dataset structure
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Gather images and assign labels
    images_domain_1 = [
        os.path.join(data_dir_1, img)
        for img in os.listdir(data_dir_1)
        if img.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
    ]
    images_domain_2_train = [str(img) for img in Path(data_dir_2).glob("*") if img.suffix in [".jpg", ".jpeg", ".png", ".gif"]]
    images_domain_2_val = [str(img) for img in Path(val_data_dir_2).glob("*") if img.suffix in [".jpg", ".jpeg", ".png", ".gif"]]

    # Validation: take equal samples from both domains
    val_images_1 = random.sample(images_domain_1, min(val_size, len(images_domain_1)))
    val_images_2 = random.sample(images_domain_2_val, min(val_size, len(images_domain_2_val)))
    val_images = [(img, 0) for img in val_images_1] + [(img, 1) for img in val_images_2]

    # Training: all images except those in validation, with domain labeling
    train_images = [(img, 0) for img in images_domain_1 if img not in val_images_1]
    train_images += [(img, 1) for img in images_domain_2_train]

    def copy_images(image_list, dest_dir):
        for idx, (img_path, label) in enumerate(image_list):
            label_dir = Path(dest_dir) / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, label_dir / f"{label}_{idx}{Path(img_path).suffix}")

    # Copy images to train and validation folders
    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)

# Usage example
data_dir_1 = "/home/ymbahram/scratch/datasets/imagenet_samples5000/"
data_dir_2 = "/home/ymbahram/scratch/pokemon/pokemon10/"
val_data_dir_2 = "/home/ymbahram/scratch/pokemon/pokemon2503/"
output_dir = "/home/ymbahram/scratch/pokemon/pokemon10classifier/"

organize_images(data_dir_1, data_dir_2, val_data_dir_2, output_dir)