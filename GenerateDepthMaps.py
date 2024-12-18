
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import os
import numpy as np
from PointCloudImageDataset import PointCloudImageDataset

import matplotlib.pyplot as plt
transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load in dataloader
pointcloud_dir = "Dataset/NewPointClouds/airplane"
image_dir = "Dataset/Images/HighResolution/train/airplane"
depth_dir = "Dataset/HighResDepthMask/airplane"
combined_dataset = PointCloudImageDataset(pointcloud_dir=pointcloud_dir, image_dir=image_dir, transform_image=transform_image)

batch_size = 16
dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=False, num_workers=0)

# Load in midas
cache_dir = "/home/hd0216/.cache/torch/hub/intel-isl_MiDaS_master" 
model_type = "DPT_Large"

midas = torch.hub.load(cache_dir, model_type, source='local')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load(cache_dir, "transforms", source='local')  # Use transforms from local cache
if model_type in ["DPT_Large", "DPT_Hybrid"]: transform = midas_transforms.dpt_transform
else: transform = midas_transforms.small_transform

# Iterate over dataset and save files
for i, (pointcloud_batch, image_batch) in enumerate(dataloader):
    print(f"Processing Batch {i + 1}:")
    image_batch = image_batch.to(device)

    print(f"Image batch shape: {image_batch.shape}")
    for batch_idx in range(image_batch.shape[0]):
        image_id = list(combined_dataset.objects.keys())[i * batch_size + batch_idx]
        image_folder = f"{depth_dir}/{image_id}"

        os.makedirs(image_folder, exist_ok=True)
        for view_idx in range(image_batch.shape[1]):
            depth_map = image_batch[batch_idx, view_idx, :, :, :]
            
            with torch.no_grad():
                prediction = midas(depth_map.unsqueeze(0))  
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_batch.shape[3:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                output = prediction.cpu().numpy()
                output_image_path = os.path.join(image_folder, f"{view_idx + 1}.png")
                plt.imsave(output_image_path, output, cmap='gray')
                print(f"Saved {output_image_path}")
    
    print(f"Finished processing batch {i + 1}")