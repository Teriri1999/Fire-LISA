import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckGeoBlock(nn.Module):
    """
    Bottleneck ResNet Block with GroupNorm.
    Structure: Proj(1x1) -> Spatial(3x3) -> Expand(1x1)
    """
    def __init__(self, dim, bottleneck_ratio=0.5):
        super().__init__()
        hidden_dim = int(dim * bottleneck_ratio)

        self.norm1 = nn.GroupNorm(32, dim, eps=1e-5)
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)

        self.norm2 = nn.GroupNorm(32, hidden_dim, eps=1e-5)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)

        self.norm3 = nn.GroupNorm(32, hidden_dim, eps=1e-5)
        self.conv3 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        
        self.act = nn.GELU()

        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.norm3(x)
        x = self.act(x)
        x = self.conv3(x)

        return shortcut + self.gamma * x


class GeoAdapter(nn.Module):
    def __init__(self, dim=1024, depth=3):
        super().__init__()
        print(f"Initializing GeoAdapter (ResNet Bottleneck, Depth={depth})...")

        self.input_norm = nn.LayerNorm(dim, eps=1e-5)

        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 32, 32))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            BottleneckGeoBlock(dim, bottleneck_ratio=0.5) for _ in range(depth)
        ])

        self.final_norm = nn.GroupNorm(32, dim, eps=1e-5)
        self.head = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_norm(x)

        B, N, C = x.shape
        H = W = int(N**0.5)

        x = x.transpose(1, 2).view(B, C, H, W)

        pos_embed = self.pos_embed.to(x.dtype)
        if pos_embed.shape[-1] != W:
            pos_embed = F.interpolate(pos_embed.float(), size=(H, W), mode='bilinear', align_corners=False).to(x.dtype)
        
        x = x + pos_embed
        x = self.blocks(x)
        x = self.final_norm(x)
        x = self.head(x)
        x = torch.clamp(x, min=-50.0, max=50.0)
        x = x.flatten(2).transpose(1, 2)
        
        return x
