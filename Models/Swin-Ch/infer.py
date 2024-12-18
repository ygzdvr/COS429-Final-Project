import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from vision_transformer_model import ImageToPointCloud

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_points):
    model = ImageToPointCloud(num_points=num_points).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def plot_pointclouds(pred_pc, actual_pc, save_path):
    fig = plt.figure(figsize=(12, 6))

    # Predicted point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2], s=1, c='r', label="Predicted")
    ax1.set_title("Predicted Point Cloud")
    ax1.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Actual point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(actual_pc[:, 0], actual_pc[:, 1], actual_pc[:, 2], s=1, c='b', label="Actual")
    ax2.set_title("Actual Point Cloud")
    ax2.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved as: {save_path}")

def run_inference(model, image_path, pc_path, transform, save_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Load the actual point cloud
    actual_pc = np.load(pc_path)

    # Run inference
    with torch.no_grad():
        pred_pc = model(image).squeeze(0).cpu().numpy()

    # Plot and save the comparison
    plot_pointclouds(pred_pc, actual_pc, save_path)


if __name__ == "__main__":
    # Paths
    model_path = "best_model_single_shot.pth"  # Trained model path
    image_path = "Dataset/Images/LowResolution/test/airplane/airplane_0629/16.png"  # Test image path
    pc_path = "Dataset/PointClouds/LowResolution/test/airplane/airplane_0628.npy"  # Corresponding point cloud path
    output_path = "pointcloud_comparison_6.png"  # Output plot path

    # Parameters
    num_points = 500  # Number of points in the point cloud

    # Load the model
    model = load_model(model_path, num_points)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Run inference and generate comparison
    run_inference(model, image_path, pc_path, transform, output_path)
