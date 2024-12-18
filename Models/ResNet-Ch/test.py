import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from vision_transformer_model import ImageToPointCloud
from data import PCD
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load the best model
best_model_path = "Models/ResNet-Ch/best_model_single_shot.pth"
model = ImageToPointCloud(num_points=500).to(device)
model.load_state_dict(torch.load(best_model_path))
model.eval()
print("Model loaded.")

# Image transforms
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Test Dataset and DataLoader
test_dataset = PCD(
    image_root="Dataset/Images/LowResolution/test/airplane",
    pc_root="Dataset/PointClouds/VeryLowResolution/test/airplane",
    transform=img_transform
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Chamfer distance function
def chamfer_distance(pc1, pc2):
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)  # Shape: (B, N, M, 3)
    dist = torch.sum(diff ** 2, dim=-1)  # Squared distances, shape: (B, N, M)

    dist1 = torch.min(dist, dim=2)[0]  # Min distance from pc1 to pc2
    dist2 = torch.min(dist, dim=1)[0]  # Min distance from pc2 to pc1

    return torch.mean(dist1) + torch.mean(dist2)

# Evaluate on the test set
def evaluate_test_set():
    test_loss = 0
    with torch.no_grad():
        for images, pcs in test_loader:
            images = images.to(device)
            pcs = pcs.to(device)

            pred_pcs = model(images)
            loss = chamfer_distance(pred_pcs, pcs)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss

# Plot point clouds
def plot_pointclouds(pred_pc, actual_pc, save_path):
    fig = plt.figure(figsize=(12, 6))

    # Predicted point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2], s=5, c='r', label="Predicted")
    ax1.set_title("Predicted Point Cloud")
    ax1.set_box_aspect([1, 1, 1])

    # Actual point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(actual_pc[:, 0], actual_pc[:, 1], actual_pc[:, 2], s=5, c='b', label="Actual")
    ax2.set_title("Actual Point Cloud")
    ax2.set_box_aspect([1, 1, 1])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved as: {save_path}")

# Main execution
def main():
    avg_test_loss = evaluate_test_set()

    # Visualize predictions for a few test samples
    for idx in range(5):  # Visualize first 5 test samples
        test_image, test_pc = test_dataset[idx]
        test_image = test_image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_pc = model(test_image).squeeze(0).cpu().numpy()

        plot_pointclouds(
            pred_pc,
            test_pc.numpy(),
            save_path=f"Models/ResNet-Ch/test_sample_{idx}.png"
        )

if __name__ == "__main__":
    main()
