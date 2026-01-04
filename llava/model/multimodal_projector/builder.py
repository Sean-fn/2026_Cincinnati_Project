import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_multimodal_projector(
    config,
    projector_type='mlp2x_gelu',
    multimodal_hidden_size=2,
    kan_hidden_dim=128,
    kan_grid_size=5,
    kan_spline_order=3,
    delay_load=False,
    **kwargs
):
    """
    Build a multimodal projector for mapping deepfake features to LLM space.

    Args:
        config: Model configuration with hidden_size attribute
        projector_type: Type of projector ('linear', 'mlp2x_gelu', 'efficient_kan', 'identity')
        multimodal_hidden_size: Input dimension (default: 2 for deepfake probabilities)
        kan_hidden_dim: Hidden dimension for KAN projector (default: 128)
        kan_grid_size: Grid size for KAN B-splines (default: 5)
        kan_spline_order: Spline order for KAN (default: 3)
        delay_load: Whether to delay loading (unused, kept for API compatibility)

    Returns:
        nn.Module: The constructed projector
    """
    if projector_type == 'linear':
        return nn.Linear(multimodal_hidden_size, config.hidden_size)

    # Efficient KAN projector for enhanced expressiveness with lower VRAM
    if projector_type == 'efficient_kan':
        from .efficient_kan import EfficientKANProjector
        return EfficientKANProjector(
            input_dim=multimodal_hidden_size,
            hidden_dim=kan_hidden_dim,
            output_dim=config.hidden_size,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(multimodal_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')