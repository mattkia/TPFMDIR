import torch

import torch.nn as nn

from typing import List
from typing import Tuple

from models.modules import ConvBlock3D
from models.modules import ConvBlock2D
from models.modules import AttentionGate3D
from models.modules import AttentionGate2D
from models.modules import SinusoidalPositionEmbeddings



class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int=2,
        out_channels: int=3, 
        down_channels: List | Tuple=(32, 32, 32), 
        up_channels: List | Tuple=(32, 32, 32), 
        time_emb_dim: int=32,
        decoder_only: bool=True,
        use_attention_gates: bool=True,
        use_se_attention: bool=False
    ) -> None:
        super().__init__()

        self.use_attn_gate = use_attention_gates
    
        # Time embedding
        self.time_pos_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.s_embedder = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.t_embedder = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList()

        encoder_time_dim = None if decoder_only else time_emb_dim
        for i in range(len(down_channels) - 1):
            block = ConvBlock2D(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                time_emb_dim=encoder_time_dim,
                up=False,
                down=True,
                use_se=use_se_attention)
            self.downs.append(block)

        # Upsample
        self.ups = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            block = ConvBlock2D(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                time_emb_dim=time_emb_dim,
                up=True,
                down=False,
                use_se=use_se_attention)
            self.ups.append(block)

        # Add attention gates
        if self.use_attn_gate:
            self.attn_gates = nn.ModuleList()

            for i in range(len(self.ups)):
                block = AttentionGate2D(
                    channels_l=down_channels[-i - 1],
                    channels_g=up_channels[i],
                    inter_channels=max(up_channels[i] // 2, 1))
                self.attn_gates.append(block)

        # Final projection
        self.output = nn.Conv2d(up_channels[-1], out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        # Embedd time
        s_ctx = self.s_embedder(self.time_pos_emb(s * 1000))
        t_ctx = self.t_embedder(self.time_pos_emb(t * 1000))
        ctx = s_ctx + t_ctx

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, ctx)
            residual_inputs.append(x)
        
        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()

            # Apply attention gate if set true
            if self.use_attn_gate:
                residual_x = self.attn_gates[i](x, residual_x)

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, ctx)

        return self.output(x)


class UNet3D(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(
        self,
        in_channels: int=2,
        out_channels: int=3, 
        down_channels: List | Tuple=(32, 32, 32), 
        up_channels: List | Tuple=(32, 32, 32), 
        time_emb_dim: int=32,
        decoder_only: bool=True,
        use_attention_gates: bool=True,
        use_se_attention: bool=False
    ) -> None:
        super().__init__()

        self.use_attn_gate = use_attention_gates
    
        # Time embedding
        self.time_pos_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.s_embedder = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.t_embedder = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial projection
        self.conv0 = nn.Conv3d(in_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList()

        encoder_time_dim = None if decoder_only else time_emb_dim
        for i in range(len(down_channels) - 1):
            block = ConvBlock3D(
                in_channels=down_channels[i],
                out_channels=down_channels[i + 1],
                time_emb_dim=encoder_time_dim,
                up=False,
                down=True,
                use_se=use_se_attention)
            self.downs.append(block)

        # Upsample
        self.ups = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            block = ConvBlock3D(
                in_channels=up_channels[i],
                out_channels=up_channels[i + 1],
                time_emb_dim=time_emb_dim,
                up=True,
                down=False,
                use_se=use_se_attention)
            self.ups.append(block)

        # Add attention gates
        if self.use_attn_gate:
            self.attn_gates = nn.ModuleList()

            for i in range(len(self.ups)):
                block = AttentionGate3D(
                    channels_l=down_channels[-i - 1],
                    channels_g=up_channels[i],
                    inter_channels=max(up_channels[i] // 2, 1))
                self.attn_gates.append(block)

        # Final projection
        self.output = nn.Conv3d(up_channels[-1], out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        # Embedd time
        s_ctx = self.s_embedder(self.time_pos_emb(s * 1000))
        t_ctx = self.t_embedder(self.time_pos_emb(t * 1000))
        ctx = s_ctx + t_ctx

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, ctx)
            residual_inputs.append(x)
        
        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()

            # Apply attention gate if set true
            if self.use_attn_gate:
                residual_x = self.attn_gates[i](x, residual_x)

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, ctx)

        return self.output(x)
