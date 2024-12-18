import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from vision_transformer_model import ImageToPointCloud
from data import PCD
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image
import os

print(torch.__version__)  # 1.9.0
# Device configuration
print(torch.cuda.is_available())  # True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # cuda

print("Model, Loss, and Optimizer")
# Model, Loss, and Optimizer
model = ImageToPointCloud(num_points=500).to(device)

# Verify
print("Adjusted positional embeddings for 224X224 input.")
print("ViT-MSE model")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

print("Image transforms")
# Image transforms
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Dataset and DataLoader")
# Dataset and DataLoader
train_dataset = PCD(
    image_root="Dataset/Images/LowResolution/train/airplane", 
    pc_root="Dataset/PointClouds/VeryLowResolution/train/airplane",
    transform=img_transform
)

# Split dataset into training and validation sets
val_size = int(0.15 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

print(f"Number of training samples: {train_size}")
print(f"Number of validation samples: {val_size}")

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

# Create directory for model archives
os.makedirs("Models/ViT-MSE/model_archive", exist_ok=True)

print("Training loop")
# Save the best model
best_validation_loss = float('inf')
best_train_loss = float('inf')
best_model_path = "Models/ViT-MSE/best_model_single_shot.pth"

# Lists to store loss values
training_losses = []
validation_losses = []

# Perform inference on a single sample
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

# Training loop
for epoch in range(50):  # Number of epochs
    print(f"Epoch {epoch}")
    model.train()
    epoch_loss = 0
    for images, pcs in train_loader:
        images = images.to(device)
        pcs = pcs.to(device)

        optimizer.zero_grad()
        pred_pcs = model(images)
        loss = criterion(pred_pcs, pcs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    training_losses.append(avg_loss)
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")
    if avg_loss < best_train_loss:
        best_train_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with training loss: {best_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, pcs in val_loader:
            images = images.to(device)
            pcs = pcs.to(device)
            
            pred_pcs = model(images)
            loss = criterion(pred_pcs, pcs)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)
    print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}")

    # Save the best model based on validation loss
    if avg_val_loss < best_validation_loss:
        best_validation_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_validation_loss:.4f}")

    # Archive model for this epoch
    epoch_model_path = f"Models/ViT-MSE/model_archive/model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), epoch_model_path)
    print(f"Model for epoch {epoch} saved at: {epoch_model_path}")

    # Inference on first entry of train_subset and val_subset
    train_sample_image, train_sample_pc = train_subset[0]
    val_sample_image, val_sample_pc = val_subset[0]

    train_sample_image = train_sample_image.unsqueeze(0).to(device)
    val_sample_image = val_sample_image.unsqueeze(0).to(device)

    with torch.no_grad():
        train_pred_pc = model(train_sample_image).squeeze(0).cpu().numpy()
        val_pred_pc = model(val_sample_image).squeeze(0).cpu().numpy()

    plot_pointclouds(train_pred_pc, train_sample_pc.numpy(), f"Models/ViT-MSE/epoch_archive/train_epoch_{epoch}.png")
    plot_pointclouds(val_pred_pc, val_sample_pc.numpy(), f"Models/ViT-MSE/epoch_archive/val_epoch_{epoch}.png")

# Plot training and validation losses
plt.figure()
plt.plot(range(50), training_losses, label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Models/ViT-MSE/training_loss_plot.png")
plt.close()

plt.figure()
plt.plot(range(50), validation_losses, label="Validation Loss")
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Models/ViT-MSE/validation_loss_plot.png")
plt.close()

plt.figure()
plt.plot(range(50), training_losses, label="Training Loss")
plt.plot(range(50), validation_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Models/ViT-MSE/training_validation_loss_plot.png")
plt.close()

# Load the best model for inference
model.load_state_dict(torch.load(best_model_path))
model.eval()