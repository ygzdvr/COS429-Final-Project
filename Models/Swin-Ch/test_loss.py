import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from vision_transformer_model import ImageToPointCloud


print(torch.__version__)  # 1.9.0
# Device configuration
print(torch.cuda.is_available())  # True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # cuda
# Chamfer distance function
def chamfer_distance(pc1, pc2):
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)  # (B, N, M, 3)
    dist = torch.sum(diff ** 2, dim=-1)  # Squared distances (B, N, M)
    dist1 = torch.min(dist, dim=2)[0]  # (B, N)
    dist2 = torch.min(dist, dim=1)[0]  # (B, M)
    return torch.mean(dist1) + torch.mean(dist2)

# Load test dataset and perform evaluation
def evaluate_model_on_test_set(model, test_image_dir, test_pc_dir, transform, num_points):
    model.eval()
    chamfer_losses = []
    
    # Iterate through all test samples
    for image_file in os.listdir(test_image_dir):
        print(os.listdir(test_image_dir))
        if not image_file.endswith('.png'):  # Adjust extension if necessary
            continue
        
        # Load and preprocess the image
        image_path = os.path.join(test_image_dir, image_file)
        pc_path = os.path.join(test_pc_dir, image_file.replace('.png', '.npy'))
        
        if not os.path.exists(pc_path):  # Ensure corresponding point cloud exists
            print(f"Missing point cloud for {image_file}")
            continue
        
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Load the actual point cloud
        actual_pc = torch.tensor(np.load(pc_path), dtype=torch.float32).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            pred_pc = model(image)
        
        # Compute Chamfer Distance
        loss = chamfer_distance(pred_pc, actual_pc)
        chamfer_losses.append(loss.item())
        
        print(f"Processed {image_file}, Chamfer Loss: {loss.item():.6f}")
    
    # Report average loss
    avg_loss = np.mean(chamfer_losses)
    print(f"Average Chamfer Distance Loss on Test Set: {avg_loss:.6f}")
    return chamfer_losses

if __name__ == "__main__":
    # Paths
    model_path = "Models/Swin-Ch/best_model_single_shot.pth"
    test_image_dir = "Dataset/Images/LowResolution/test/airplane"
    test_pc_dir = "Dataset/PointClouds/VeryLowResolution/test/airplane"
    
    # Parameters
    num_points = 500
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load the model
    model = ImageToPointCloud(num_points=num_points).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluate the model
    chamfer_losses = evaluate_model_on_test_set(model, test_image_dir, test_pc_dir, transform, num_points)
