import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

from .geo_adapter import GeoAdapter
from .geo_encoder import GeoEncoder

class GeoAlignmentModel(nn.Module):
    def __init__(self, 
                 clip_model_name="openai/clip-vit-large-patch14", 
                 geo_model_name="convnext_tiny",
                 adapter_dim=1024):
        super().__init__()

        print(f"Loading CLIP: {clip_model_name}...")
        self.vision_tower = CLIPVisionModel.from_pretrained(clip_model_name)
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.vision_tower.eval()

        print(f"Loading GeoEncoder (Teacher): {geo_model_name}...")
        self.geo_tower = GeoEncoder(
            output_dim=adapter_dim, 
            model_name=geo_model_name, 
            freeze_backbone=True
        )
        
        for param in self.geo_tower.parameters():
            param.requires_grad = False
        self.geo_tower.eval()

        print("Initializing GeoAdapter (Student)...")
        self.geo_adapter = GeoAdapter(dim=adapter_dim, depth=3)

        for param in self.geo_adapter.parameters():
            param.requires_grad = True

        print("⚡ Adjusting Adapter initialization for fast convergence...")
        for m in self.geo_adapter.modules():
            if hasattr(m, 'gamma'):
                nn.init.constant_(m.gamma, 1e-4)

    def forward(self, rgb_images, geo_images):    
        with torch.no_grad():
            clip_outputs = self.vision_tower(rgb_images, output_hidden_states=True)
            clip_feat = clip_outputs.hidden_states[-2]
            clip_feat_patches = clip_feat[:, 1:, :] 
            
            n_tokens = clip_feat_patches.shape[1]
            grid_size = int(n_tokens ** 0.5)

        with torch.no_grad():
            # geo_images: [B, 6, H, W]  S2 pre(3ch) + post(3ch)
            geo_feat_real = self.geo_tower(
                geo_images,
                target_size=(grid_size, grid_size)
            )
            if torch.isnan(geo_feat_real).any():
                geo_feat_real = torch.nan_to_num(geo_feat_real, nan=0.0)

        geo_feat_pred = self.geo_adapter(clip_feat_patches)

        target = geo_feat_real.detach().flatten(0, 1)
        pred = geo_feat_pred.flatten(0, 1)

        target = torch.nan_to_num(target, nan=0.0, posinf=100.0, neginf=-100.0)
        target = torch.clamp(target, min=-100, max=100)

        loss = F.smooth_l1_loss(pred, target, beta=1.0)

        return loss
