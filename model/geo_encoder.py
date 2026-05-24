import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GeoEncoder(nn.Module):
    def __init__(self, output_dim=1024, model_name='convnext_tiny', freeze_backbone=True, pretrained_path=None):
        super().__init__()
        print(f"Loading GeoEncoder: {model_name}...")

        use_timm_pretrained = (pretrained_path is None)
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_timm_pretrained,
            in_chans=6,   # S2 pre(3ch) + post(3ch)
            num_classes=0,
            global_pool=''
        )

        if pretrained_path is not None:
            print(f"Loading local weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Local weights loaded. Missing keys: {len(msg.missing_keys)}")

        self.backbone_dim = self.backbone.num_features

        self.adapter = nn.Sequential(
            nn.Conv2d(self.backbone_dim, output_dim, kernel_size=1),
            nn.GELU()
        )
        nn.init.xavier_normal_(self.adapter[0].weight)
        nn.init.constant_(self.adapter[0].bias, 0)

        self.layernorm = nn.LayerNorm(output_dim)

        if freeze_backbone:
            print("Freezing GeoEncoder Backbone (Keeping ImageNet weights)...")
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.adapter.parameters():
                param.requires_grad = True

            self.backbone.eval()

    def forward(self, x, target_size=None):
        with torch.no_grad():
            features = self.backbone(x)

        features = self.adapter(features)

        if target_size is not None:
            original_dtype = features.dtype
            features = features.to(torch.float32)

            features = F.interpolate(
                features, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )

            features = features.to(original_dtype)

        features = features.flatten(2).transpose(1, 2)
        features = self.layernorm(features)
        
        return features

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
