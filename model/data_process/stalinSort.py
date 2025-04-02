# dataset reducer

import torch
import os

# Paths for the original dataset
current_path = os.path.dirname(os.path.abspath(__file__))
coords_path = os.path.join(current_path, '../registry/dataset', 'dataset_coords.pt')
images_path = os.path.join(current_path, '../registry/dataset', 'dataset_images.pt')

# Paths for the new reduced dataset
reduced_coords_path = os.path.join(current_path, '../registry/dataset', 'microset_coords.pt')
reduced_images_path = os.path.join(current_path, '../registry/dataset', 'microset_images.pt')

print("Before loading dataset")

# Load the original dataset
coords = torch.load(coords_path)
images = torch.load(images_path)

print("After loading dataset")

# Calculate the number of samples (percents% rounded up)
percents = 5
num_samples = max(1, -(-len(coords) * percents // 100))

print(f"num_samples: {num_samples}")
print("Lets do [:num_samples]")

# Select the first percents% of the dataset
reduced_coords = coords[:num_samples]
reduced_images = images[:num_samples]

print("Before torch.save")

# Save the reduced dataset
torch.save(reduced_coords, reduced_coords_path)
torch.save(reduced_images, reduced_images_path)

print(f'Successfully created reduced dataset with {num_samples} samples.')
