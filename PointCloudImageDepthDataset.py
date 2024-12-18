import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import os
import numpy as np
from PointCloudImageDataset import PointCloudImageDataset
import matplotlib.pyplot as plt

# Dataset class
class PointCloudImageDepthDataset(Dataset):
    def __init__(self, pointcloud_dir, image_dir, depth_dir, transform_image=None, transform_depth=None, num_points=1000):

        # Load directories for importing files
        self.pointcloud_dir = pointcloud_dir
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform_image = transform_image
        self.transform_depth = transform_depth
        self.num_points = num_points

        self.objects = {}

        # Check for image / pointcloud file pairings
        for file in os.listdir(self.pointcloud_dir):
            if file.endswith(".txt"):
                pointcloud_path = os.path.join(self.pointcloud_dir, file)
                object_name = file.split("_pointcloud")[0]

                images_path = os.path.join(self.image_dir, object_name)
                images_paths = []
                if os.path.isdir(images_path):
                    for i in range(1, 41):
                        image_path = os.path.join(images_path, f"{i}.png")
                        if os.path.exists(image_path): images_paths.append(image_path)

                depths_path = os.path.join(self.depth_dir, object_name)
                depths_paths = []
                if os.path.isdir(depths_path):
                    for i in range(1, 41):
                        depth_path = os.path.join(depths_path, f"{i}.png")
                        if os.path.exists(depth_path): depths_paths.append(depth_path)

                # Chech that 40 images exist before adding to the objects list
                if (len(depths_paths) == 40 and len(images_paths) == 40): self.objects[object_name] = [pointcloud_path, images_paths, depths_paths]
                else: print(f"Images or depths not found for object {object_name}")

        print("Loaded", len(self.objects), "objects")

    # Get size of dataset
    def __len__(self):
        return len(self.objects)
    
    # Get n_th item from dataset
    def __getitem__(self, idx):
        object_name = list(self.objects.keys())[idx]
        pointcloud_path = self.objects[object_name][0]
        pointcloud = self.load_pointcloud(pointcloud_path)

        # Check for both image and pointcloud paths existing
        image_paths = self.objects[object_name][1]
        if image_paths is None: raise ValueError(f"Images not found for object {object_name}")
        images = [self.load_image(image_path) for image_path in image_paths]

        # Check for depth maps paths existing
        depth_paths = self.objects[object_name][2]
        if depth_paths is None: raise ValueError(f"Depths not found for object {object_name}")
        depths =[self.load_depth(depth_path) for depth_path in depth_paths]

        # Apply transformations to images and depth maps
        if self.transform_image: images = [self.transform_image(image) for image in images]
        if self.transform_depth: depths = [self.transform_depth(depth) for depth in depths]

        # Stack images into a batch (shape [40, 3, 224, 224])
        image_batch = torch.stack(images)
        depth_batch = torch.stack(depths)

        return pointcloud, image_batch, depth_batch
    
    # Load numpy file into torch tensor
    def load_pointcloud(self, pointcloud_path):
        pointcloud = np.loadtxt(pointcloud_path)
        if pointcloud.shape[0] >= self.num_points: pointcloud = pointcloud[:self.num_points]
        else: pointcloud = np.pad(pointcloud, ((0, self.num_points - pointcloud.shape[0]), (0, 0)), mode='constant')
        return torch.tensor(pointcloud, dtype=torch.float32)
    
    # Load image file in PIL format with RGB
    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    # Load depth file in PIL format with RGB 
    def load_depth(self, depth_path):
        depth = Image.open(depth_path).convert("RGB")
        return depth

# Transformations to apply to loaded images
transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformations to apply to depth images
transform_depth = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Directories of files
pointcloud_dir = "Dataset/NewPointClouds/airplane"
image_dir = "Dataset/Images/LowResolution/train/airplane"
depth_dir = "Dataset/DepthMaps/airplane"

combined_dataset = PointCloudImageDepthDataset(pointcloud_dir=pointcloud_dir, image_dir=image_dir, depth_dir=depth_dir, transform_image=transform_image, transform_depth=transform_depth)
dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True, num_workers=0)

# Using dataloader
for i, (pointcloud_batch, image_batch, depth_batch) in enumerate(dataloader):
    print(f"Batch {i + 1}:")
    print(f"Point Cloud Batch Shape: {pointcloud_batch.shape}")     # Should be [batch_size, 1000, 3]
    print(f"Image Batch Shape: {image_batch.shape}")                # Should be [batch_size, 40, 3, 224, 224]
    print(f"Depth Batch Shape: {depth_batch.shape}")                # Should be [batch_size, 40, 3, 224, 224]