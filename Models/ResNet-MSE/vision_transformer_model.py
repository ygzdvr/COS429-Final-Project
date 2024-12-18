import torch
import torch.nn as nn
import torchvision.models as models

class ImageToPointCloud(nn.Module):
    def __init__(self, num_points=500):
        super().__init__()
        # Load a ResNet model (e.g., ResNet-50)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the classifier head (fully connected layer)
        self.resnet.fc = nn.Identity()

        # Feature dimension from ResNet-50 (depends on the ResNet version)
        feature_dim = 2048
        self.num_points = num_points
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )
        
    def forward(self, x):
        # x is [B, C, H, W]
        features = self.resnet(x)  # [B, feature_dim]
        pc = self.fc(features)  # [B, num_points * 3]
        pc = pc.view(-1, self.num_points, 3)
        return pc

# Example usage
if __name__ == "__main__":
    model = ImageToPointCloud(num_points=500)
    input_tensor = torch.randn(8, 3, 224, 224)  # Batch of 8 images, 3 channels, 224x224
    output = model(input_tensor)  # Output point cloud with shape [8, 500, 3]
    print(output.shape)
