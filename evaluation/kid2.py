import torch
import lpips  # LPIPS library
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# Initialize LPIPS loss (set model='net' for the pretrained AlexNet, VGG, etc.)
lpips_fn = lpips.LPIPS(net='alex')

# Assuming you have a pre-trained generator model
# generator_model is a PyTorch model that generates images
def generate_images(generator_model, num_images=1000, latent_dim=128, device='cuda'):
    generator_model.eval()  # Set model to evaluation mode
    latent_vectors = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        generated_images = generator_model(latent_vectors)
    return generated_images

# Assuming target_images is a tensor of shape (10, C, H, W) containing the 10 target domain images
# The images should be on the same device as the generator (e.g., 'cuda')
def assign_to_clusters(generated_images, target_images, lpips_fn):
    num_target_images = target_images.size(0)
    num_generated_images = generated_images.size(0)

    # Initialize cluster assignments
    cluster_assignments = np.zeros(num_generated_images)

    # Compute LPIPS distances and assign each generated image to the closest target image (cluster)
    for i in range(num_generated_images):
        min_distance = float('inf')
        best_cluster = -1
        for j in range(num_target_images):
            dist = lpips_fn(generated_images[i].unsqueeze(0), target_images[j].unsqueeze(0)).item()
            if dist < min_distance:
                min_distance = dist
                best_cluster = j
        cluster_assignments[i] = best_cluster

    return cluster_assignments

# Compute intra-cluster LPIPS
def compute_intra_cluster_lpips(generated_images, cluster_assignments, lpips_fn):
    unique_clusters = np.unique(cluster_assignments)
    intra_cluster_lpips_scores = []

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        cluster_images = generated_images[cluster_indices]

        # Compute pairwise LPIPS distances within the cluster
        pairwise_distances = []
        for i in range(len(cluster_images)):
            for j in range(i + 1, len(cluster_images)):
                dist = lpips_fn(cluster_images[i].unsqueeze(0), cluster_images[j].unsqueeze(0)).item()
                pairwise_distances.append(dist)

        # Average pairwise LPIPS distance for this cluster
        if pairwise_distances:
            avg_lpips = np.mean(pairwise_distances)
            intra_cluster_lpips_scores.append(avg_lpips)

    # Average LPIPS across all clusters
    return np.mean(intra_cluster_lpips_scores)

# Example usage
device = 'cuda'
generator_model = ...  # Load your pre-trained generator model
target_images = ...  # Load your target domain images (10 images)

# Generate 1000 images
generated_images = generate_images(generator_model, num_images=1000, latent_dim=128, device=device)

# Assign generated images to clusters based on LPIPS distance to target images
cluster_assignments = assign_to_clusters(generated_images, target_images, lpips_fn)

# Compute intra-cluster LPIPS
intra_cluster_lpips = compute_intra_cluster_lpips(generated_images, cluster_assignments, lpips_fn)

print(f"Intra-cluster LPIPS score: {intra_cluster_lpips}")