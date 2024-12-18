import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
    
class PCD(Dataset):
    def __init__(self, image_root, pc_root, transform=None):
        self.transform = transform
        # Each object_name corresponds to a single 3D model
        self.objects = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]

        self.samples = []
        for obj in self.objects:
            img_dir = os.path.join(image_root, obj)
            if os.path.exists(img_dir):
                # images for this object
                img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
                # corresponding pointcloud file
                pc_file = os.path.join(pc_root, obj + ".npy")

                if os.path.exists(pc_file):
                    for img_file in img_files:
                        self.samples.append((img_file, pc_file))
            else:
                print("Image directory does not exist")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, pc_file = self.samples[idx]
        image = Image.open(img_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        pointcloud = np.load(pc_file)
        pointcloud = torch.from_numpy(pointcloud).float()

        return image, pointcloud