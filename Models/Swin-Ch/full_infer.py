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

def plot_comparisons(image, pred_pc_epoch1, pred_pc_best, actual_pc, save_path):
    fig = plt.figure(figsize=(18, 6))

    # Input image
    ax0 = fig.add_subplot(141)
    ax0.imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert CHW to HWC
    ax0.set_title("Input Image")
    ax0.axis('off')

    # Predicted point cloud (epoch 1)
    ax1 = fig.add_subplot(142, projection='3d')
    ax1.scatter(pred_pc_epoch1[:, 0], pred_pc_epoch1[:, 1], pred_pc_epoch1[:, 2], s=5, c='r')
    ax1.set_title("Prediction (Epoch 1)")
    ax1.set_box_aspect([1, 1, 1])

    # Predicted point cloud (best model)
    ax2 = fig.add_subplot(143, projection='3d')
    ax2.scatter(pred_pc_best[:, 0], pred_pc_best[:, 1], pred_pc_best[:, 2], s=5, c='g')
    ax2.set_title("Prediction (Best Model)")
    ax2.set_box_aspect([1, 1, 1])

    # Actual point cloud
    ax3 = fig.add_subplot(144, projection='3d')
    ax3.scatter(actual_pc[:, 0], actual_pc[:, 1], actual_pc[:, 2], s=5, c='b')
    ax3.set_title("Ground Truth")
    ax3.set_box_aspect([1, 1, 1])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved as: {save_path}")

def run_inference(model_epoch1, model_best, image_path, pc_path, transform, save_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Load the actual point cloud
    actual_pc = np.load(pc_path)

    # Run inference with both models
    with torch.no_grad():
        pred_pc_epoch1 = model_epoch1(image_tensor).squeeze(0).cpu().numpy()
        pred_pc_best = model_best(image_tensor).squeeze(0).cpu().numpy()

    # Plot and save the comparison
    plot_comparisons(transform(image), pred_pc_epoch1, pred_pc_best, actual_pc, save_path)

if __name__ == "__main__":
    # Paths
    model_path_epoch1 = "Models/Swin-Ch/model_archive/model_epoch_0.pth"  # 1st epoch model path
    model_path_best = "Models/Swin-Ch/best_model_single_shot.pth"  # Best model path
    image_path = "Dataset/Images/LowResolution/test/airplane/airplane_0681/13.png"  # Test image path
    pc_path = "Dataset/PointClouds/VeryLowResolution/test/airplane/airplane_0681.npy"  # Corresponding point cloud path
    output_path = "Models/Swin-Ch/pointcloud_comparison_full4.png"  # Output plot path

    # Parameters
    num_points = 500  # Number of points in the point cloud

    # Load the models
    model_epoch1 = load_model(model_path_epoch1, num_points)
    model_best = load_model(model_path_best, num_points)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Run inference and generate comparison
    run_inference(model_epoch1, model_best, image_path, pc_path, transform, output_path)