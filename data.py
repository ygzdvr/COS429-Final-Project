import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import os
import numpy as np
import random

# Dataset class
class PointCloudImageDataset(Dataset):
    def __init__(self, pointcloud_dir, image_dir, transform_image=None, num_points=500):

        # Load directories for importing files
        self.pointcloud_dir = pointcloud_dir
        self.image_dir = image_dir
        self.transform_image = transform_image
        self.num_points = num_points

        self.objects = []

        # Check for image / pointcloud file pairings
        for file in os.listdir(self.pointcloud_dir):
            if file.endswith(".npy"):
                pointcloud_path = os.path.join(self.pointcloud_dir, file)
                object_name = file.split(".")[0]

                images_path = os.path.join(self.image_dir, object_name)
                images_paths = []
                if os.path.isdir(images_path):
                    for i in range(1, 41):
                        image_path = os.path.join(images_path, f"{i}.png")
                        if os.path.exists(image_path):
                            images_paths.append(image_path)

                # Chech that 40 images exist before adding to the objects list
                if len(images_paths) == 40:
                    random.shuffle(images_paths)
                    for i in range(0, 40, 10):
                        self.objects.append((pointcloud_path, images_paths[i:i+10]))

        print("Loaded", len(self.objects), "entries (including splits)")

    # Get size of dataset
    def __len__(self):
        return len(self.objects)
    
    # Get n_th item from dataset
    def __getitem__(self, idx):
        pointcloud_path, image_paths = self.objects[idx]
        pointcloud = np.load(pointcloud_path)

        # Check for both image and pointcloud paths existing
        if image_paths is None: raise ValueError(f"Images not found for object at index {idx}")
        images = [self.load_image(image_path) for image_path in image_paths]
        if self.transform_image: images = [self.transform_image(image) for image in images]
        image_batch = torch.stack(images)

        return pointcloud, image_batch
    
    # Load image from image_path in PIL form to RGB
    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

# Transformations to apply to loaded images
transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pointcloud_dir = "Dataset/PointClouds/VeryLowResolution/train/airplane"
image_dir = "Dataset/Images/LowResolution/train/airplane"
combined_dataset = PointCloudImageDataset(pointcloud_dir=pointcloud_dir, image_dir=image_dir, transform_image=transform_image)

# Using dataloader
dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True, num_workers=0)
for i, (pointcloud_batch, image_batch) in enumerate(dataloader):
    print(f"Batch {i + 1}:")
    print(f"Point Cloud Batch Shape: {pointcloud_batch.shape}")     # Should be [batch_size, num_points, 3]
    print(f"Image Batch Shape: {image_batch.shape}")                # Should be [batch_size, 10, 3, 224, 224]