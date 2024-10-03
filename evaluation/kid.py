import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from scipy import linalg
import warnings

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use a pretrained Inception v3 model and move it to the GPU if available
inception_model = inception_v3(weights=None, transform_input=False)
INCEPTION_V3_PATH = "/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/evaluation/inception_v3_google-1a9a5a14.pth"
# Load the weights from the local file
state_dict = torch.load(INCEPTION_V3_PATH)
inception_model.load_state_dict(state_dict)
# Set the model to evaluation mode and move it to the device
inception_model.fc = torch.nn.Identity()  # Remove the final classification layer
inception_model.eval()
inception_model.eval().to(device)





# Function to extract features using Inception (on a batch)
def get_inception_features(images):
    images = images.to(device)  # Move images to GPU
    with torch.no_grad():
        features = inception_model(images)
    return features


# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
#_______________________ KID COMPUTATION ____________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________


# Define a polynomial kernel for KID calculation
def polynomial_kernel(x, y, degree=3, coef0=1):
    return (x @ y.T + coef0) ** degree

# Function to compute the KID score incrementally without storing all features in memory
def compute_kid_incrementally(real_loader, generated_loader):
    k_rr_total, k_gg_total, k_rg_total = 0, 0, 0
    count_real, count_generated = 0, 0
    
    # Loop through real and generated datasets in batches

    for real_images, generated_images in zip(real_loader, generated_loader):
        real_features = get_inception_features(real_images)  # Extract features for real images
        generated_features = get_inception_features(generated_images)  # Extract features for generated images
        
        # Move features to CPU for kernel computation (optional, can stay on GPU if memory allows)
        real_features = real_features.to('cpu')
        generated_features = generated_features.to('cpu')
        
        # Update the counts of images
        m = real_features.size(0)
        n = generated_features.size(0)
        count_real += m
        count_generated += n

        # Compute the kernel terms for real-real, generated-generated, and real-generated
        k_rr_total += polynomial_kernel(real_features, real_features).sum().item() - torch.diagonal(polynomial_kernel(real_features, real_features)).sum().item()
        k_gg_total += polynomial_kernel(generated_features, generated_features).sum().item() - torch.diagonal(polynomial_kernel(generated_features, generated_features)).sum().item()
        k_rg_total += polynomial_kernel(real_features, generated_features).sum().item()
    
    # Final KID value after accumulating over batches
    kid_value = (k_rr_total / (count_real * (count_real - 1)) + 
                 k_gg_total / (count_generated * (count_generated - 1)) - 
                 2 * k_rg_total / (count_real * count_generated))
    
    return kid_value



# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
#_______________________ FID COMPUTATION ____________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________



# FID calculation function with improved numerical stability
def compute_fid_incrementally_with_stability(real_acts, generated_acts, eps=1e-6):
    """ 
    Calculate FID between two sets of activations, ensuring numerical stability
    """
    # Calculate mean and covariance for real and generated activations
    mu_real, sigma_real = np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_acts, axis=0), np.cov(generated_acts, rowvar=False)
    
    # Mean difference
    diff = mu_real - mu_gen

    # Compute square root of the product of covariance matrices
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
    
    # Check if covmean contains any NaNs or infinities (numerical issues)
    if not np.isfinite(covmean).all():
        warnings.warn(f"Covariance product contains non-finite values. Adding epsilon to diagonal of covariance matrices.")
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_gen + offset))
    
    # If covmean contains imaginary parts, remove them (as done in TensorFlow code)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            warnings.warn("Covmean has significant imaginary components, taking real part.")
        covmean = covmean.real
    
    # Compute the trace of the sqrt of the covariance product
    tr_covmean = np.trace(covmean)
    
    # Final FID formula
    fid_value = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * tr_covmean
    return fid_value

# Function to extract activations using the InceptionV3 model
def extract_inception_activations(dataloader, inception_model, device='cuda'):
    """ Extracts features using the inception model """
    inception_model = inception_model.to(device)
    activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Get inception features
            features = inception_model(batch)
            activations.append(features.cpu().numpy())  # Store on CPU as numpy for FID/KID
    
    return np.concatenate(activations, axis=0)




# _____________________________________________________________
# _____________________________________________________________
# ___________________________ NPZ DATASSET ____________________
# _____________________________________________________________
# _____________________________________________________________



class NPZImageDataset(Dataset):
    def __init__(self, npz_file_path, transform=None, key='images'):
        # Load the .npz file
        self.data = np.load(npz_file_path)
        # Assuming the images are stored under the key 'images'
        self.images = self.data[key]
        # Optional transformation pipeline (e.g., normalization, resizing, etc.)
        self.transform = transform

    def __len__(self):
        # Return the number of samples (images)
        return len(self.images)

    def __getitem__(self, idx):
        # Get image at index `idx`
        image = self.images[idx]
        
        # Convert the image from NumPy array to PyTorch tensor
        image = torch.from_numpy(image).float()

        # If the image has shape [H, W, C], permute to [C, H, W] (PyTorch expects channels first)
        if image.ndimension() == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)  # Rearrange from [H, W, C] to [C, H, W]

        # Apply any transformations (if specified)
        if self.transform:
            image = self.transform(image)

        return image



transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to 299x299 (for InceptionV3, for example)
 ])

# Create the dataset
ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'
sample_path = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/0/samples/samples_75.npz'
ref_dataset = NPZImageDataset(npz_file_path=ref_path, transform=transform, key='images')
sample_dataset = NPZImageDataset(npz_file_path=sample_path, transform=transform, key='arr_0')


# Assuming you have DataLoaders for your real and generated images
real_loader = DataLoader(ref_dataset, batch_size=64, shuffle=False)
generated_loader = DataLoader(sample_dataset, batch_size=64, shuffle=False)

# Extract activations for real and generated datasets
real_activations = extract_inception_activations(real_loader, inception_model)
generated_activations = extract_inception_activations(generated_loader, inception_model)

# Compute FID with improved numerical stability
fid_score = compute_fid_incrementally_with_stability(real_activations, generated_activations)
print(f"FID Score: {fid_score}")