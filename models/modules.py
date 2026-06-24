import math
import torch

import torch.nn as nn
import torch.nn.functional as F



def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    out = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return out



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int
    ) -> None:
        super().__init__()

        self.dim = dim

    def forward(
        self,
        time
    ) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class SEBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int=8
    ) -> None:
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)

        return x * w


class SEBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 8
    ) -> None:
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)

        return x * w


class AttentionGate3D(nn.Module):
    def __init__(
        self,
        channels_l: int,
        channels_g: int,
        inter_channels: int=None
    ) -> None:
        super().__init__()

        if inter_channels is None:
            inter_channels = max(channels_l // 2, 1)
        
        self.weights_g = nn.Sequential(
            nn.Conv3d(channels_g, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )

        self.weights_x = nn.Sequential(
            nn.Conv3d(channels_l, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        g: torch.Tensor,
        l: torch.Tensor
    ) -> torch.Tensor:
        """
        g: decoded features
        l: encoder skiped features
        """

        attention = self.psi(self.weights_g(g) + self.weights_x(l))

        out = l * attention

        return out


class AttentionGate2D(nn.Module):
    def __init__(
        self,
        channels_l: int,
        channels_g: int,
        inter_channels: int=None
    ) -> None:
        super().__init__()

        if inter_channels is None:
            inter_channels = max(channels_l // 2, 1)
        
        self.weights_g = nn.Sequential(
            nn.Conv2d(channels_g, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.weights_x = nn.Sequential(
            nn.Conv2d(channels_l, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        g: torch.Tensor,
        l: torch.Tensor
    ) -> torch.Tensor:
        """
        g: decoded features
        l: encoder skiped features
        """

        attention = self.psi(self.weights_g(g) + self.weights_x(l))

        out = l * attention

        return out


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int=None,
        time_emb_dim: int=None,
        up: bool=False,
        down: bool=True,
        use_se: bool=False
    ) -> None:
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.act = F.silu

        if time_emb_dim is not None:
            self.time_mlp =  nn.Linear(time_emb_dim, out_channels)

        if up:
            # Upsampling (decoder part)
            up_channels = in_channels + skip_channels if skip_channels else 2 * in_channels
            self.conv1 = nn.Conv3d(up_channels, out_channels, 3, padding=1)
            self.transform = nn.Sequential(nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                           nn.Upsample(scale_factor=2, mode='trilinear'))
        elif down:
            # Downsampling (encoder part)
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
            if down:
                self.transform = nn.Conv3d(out_channels, out_channels, 4, 2, 1)
            else:
                self.transform = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(out_channels)
        self.bnorm2 = nn.BatchNorm3d(out_channels)

        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock3D(out_channels)

    def forward(
        self, 
        x:torch.Tensor,
        ctx: torch.Tensor
    ) -> torch.Tensor:
        # First Conv
        h = self.bnorm1(self.act(self.conv1(x)))    
        
        if self.time_emb_dim is not None:
            ctx = self.time_mlp(ctx)
            
            # Add time channel
            h = h + ctx[..., None, None, None]

        # Second Conv
        h = self.bnorm2(self.act(self.conv2(h)))

        if self.use_se:
            h = self.se_block(h)
    
        # Down or Upsample
        out = self.act(self.transform(h))
    
        return out


class ConvBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int=None,
        time_emb_dim: int=None,
        up: bool=False,
        down: bool=True,
        use_se: bool=False
    ) -> None:
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.act = F.silu

        if time_emb_dim is not None:
            self.time_mlp =  nn.Linear(time_emb_dim, out_channels)

        if up:
            # Upsampling (decoder part)
            up_channels = in_channels + skip_channels if skip_channels else 2 * in_channels
            self.conv1 = nn.Conv2d(up_channels, out_channels, 3, padding=1)
            self.transform = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                           nn.Upsample(scale_factor=2, mode='bilinear'))
        elif down:
            # Downsampling (encoder part)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            if down:
                self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            else:
                self.transform = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock2D(out_channels)

    def forward(
        self, 
        x:torch.Tensor,
        ctx: torch.Tensor
    ) -> torch.Tensor:
        # First Conv
        h = self.bnorm1(self.act(self.conv1(x)))    
        
        if self.time_emb_dim is not None:
            ctx = self.time_mlp(ctx)
            
            # Add time channel
            h = h + ctx[..., None, None]

        # Second Conv
        h = self.bnorm2(self.act(self.conv2(h)))

        if self.use_se:
            h = self.se_block(h)
    
        # Down or Upsample
        out = self.act(self.transform(h))
    
        return out
