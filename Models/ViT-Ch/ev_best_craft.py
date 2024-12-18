import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from chamferdist import ChamferDistance
from vision_transformer_model import ImageToPointCloud  # Import your model class
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
def load_model(model_path, num_points=500):
    model = ImageToPointCloud(num_points=num_points).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Find and display the lowest scoring image comparison
def find_lowest_scoring_image(image_root, pc_root, model, transform):
    chamfer_loss = ChamferDistance()
    lowest_loss = float("inf")
    lowest_result = None

    # Traverse through all objects
    for obj_dir in os.listdir(image_root):
        obj_image_dir = os.path.join(image_root, obj_dir)
        obj_pc_path = os.path.join(pc_root, obj_dir + ".npy")

        # Skip if corresponding point cloud does not exist
        if not os.path.exists(obj_pc_path):
            continue

        # Load the actual point cloud
        actual_pc = np.load(obj_pc_path)  # Shape: [N, 3]
        actual_pc_tensor = torch.from_numpy(actual_pc).float().unsqueeze(0).to(device)

        # Traverse through all images for this object
        for image_file in os.listdir(obj_image_dir):
            if not image_file.endswith(".png"):
                continue

            # Load and transform the image
            image_path = os.path.join(obj_image_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict the point cloud
            with torch.no_grad():
                pred_pc_tensor = model(image_tensor)

            # Compute Chamfer Distance loss
            loss = chamfer_loss(pred_pc_tensor, actual_pc_tensor)
            loss_value = loss.item()

            # Update the lowest loss
            if loss_value < lowest_loss:
                lowest_loss = loss_value
                lowest_result = (image_path, pred_pc_tensor.cpu().squeeze(0).numpy(), actual_pc)

    # Plot the lowest-scoring image comparison
    if lowest_result:
        image_path, pred_pc, actual_pc = lowest_result
        print(f"Lowest loss: {lowest_loss:.4f}, Image: {image_path}")

        # Plot comparison
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(Image.open(image_path))
        ax1.set_title("Input Image")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2], s=1, c='r', label="Predicted")
        ax2.scatter(actual_pc[:, 0], actual_pc[:, 1], actual_pc[:, 2], s=1, c='b', label="Actual")
        ax2.set_title(f"Loss: {lowest_loss:.4f}")
        ax2.legend()
        ax2.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        plt.tight_layout()
        plt.show()
    else:
        print("No valid image and point cloud pair found.")

# Example usage
if __name__ == "__main__":
    # Paths
    image_root = "Dataset/Images/LowResolution/train/airplane"  # Path to the images
    pc_root = "Dataset/PointClouds/VeryLowResolution/train/airplane"  # Path to the point clouds
    model_path = "best_model_single_shot.pth"  # Path to the saved model

    # Parameters
    num_points = 500
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the model
    model = load_model(model_path, num_points)

    # Find and display the lowest-scoring image comparison
    find_lowest_scoring_image(image_root, pc_root, model, transform)
