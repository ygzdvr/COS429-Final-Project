import torch
import torch.nn as nn
import torchvision.models as models

class ImageToPointCloud(nn.Module):
    def __init__(self, num_points=500):
        super().__init__()
        # Load a ViT model
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Remove the classifier head
        self.vit.heads = nn.Identity()
        
        # Now self.vit outputs a feature vector of size 768 by default
        feature_dim = self.vit.hidden_dim
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )
        
    def forward(self, x):
        # x is [B, C, H, W]
        features = self.vit(x)  # [B, feature_dim]
        pc = self.fc(features)  # [B, num_points*3]
        pc = pc.view(-1, self.num_points, 3)
        return pc
