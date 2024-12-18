import torch
import torch.nn as nn
import torchvision.models as models

class ImageToPointCloud(nn.Module):
    def __init__(self, num_points=500):
        super().__init__()
        # Load a Swin Transformer model
        self.swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

        # Remove the classifier head
        self.swin.head = nn.Identity()
        
        # Now self.swin outputs a feature vector of size 768 by default
        feature_dim = 768  # Feature dimension from Swin-T base model
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )
        
    def forward(self, x):
        # x is [B, C, H, W]
        features = self.swin(x)  # [B, feature_dim]
        pc = self.fc(features)  # [B, num_points*3]
        pc = pc.view(-1, self.num_points, 3)
        return pc
