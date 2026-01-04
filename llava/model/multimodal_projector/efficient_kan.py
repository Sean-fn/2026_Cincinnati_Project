"""
Efficient KAN (Kolmogorov-Arnold Network) Projector for M2F2-Det

This implementation is based on the efficient-kan approach which reformulates KAN
computation for better GPU efficiency and FP16 compatibility.

Key features:
- B-spline basis functions computed efficiently
- Compatible with mixed precision training (FP16/BF16)
- Designed for the deepfake projector use case (2 -> hidden -> 4096)

Reference: https://github.com/Blealtan/efficient-kan
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class EfficientKANLinear(nn.Module):
    """
    Efficient implementation of a single KAN layer.

    Combines a base linear transformation with learnable B-spline activation functions.
    The output is: base_activation(x) @ base_weight + spline(x) @ spline_weight

    Args:
        in_features: Input dimension
        out_features: Output dimension
        grid_size: Number of grid intervals for B-splines (default: 5)
        spline_order: Order of B-spline (default: 3 for cubic)
        scale_noise: Noise scale for spline weight initialization
        scale_base: Scale for base weight initialization
        scale_spline: Scale for spline scaler initialization
        base_activation: Activation function for base path (default: SiLU)
        grid_range: Range for the spline grid (default: [-1, 1])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type = nn.SiLU,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        # Compute grid step size
        h = (grid_range[1] - grid_range[0]) / grid_size

        # Create grid buffer (non-trainable)
        # Grid extends beyond range to support spline computation at boundaries
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1).float() * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Base linear transformation weight
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Spline coefficients: [out_features, in_features, num_coeffs]
        # num_coeffs = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        # Learnable scaling for spline outputs
        self.spline_scaler = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Base activation function
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        # Base weight: Kaiming initialization
        nn.init.kaiming_uniform_(
            self.base_weight,
            a=math.sqrt(5) * self.scale_base
        )

        # Spline scaler: Kaiming initialization
        nn.init.kaiming_uniform_(
            self.spline_scaler,
            a=math.sqrt(5) * self.scale_spline
        )

        # Spline weights: Small noise initialization
        with torch.no_grad():
            noise = (
                (torch.rand_like(self.spline_weight) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.copy_(noise)

    def compute_bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions for input x.

        Uses the Cox-de Boor recursion formula, implemented efficiently for GPU.

        Args:
            x: Input tensor of shape [batch_size, in_features]

        Returns:
            Basis tensor of shape [batch_size, in_features, num_coeffs]
        """
        # x: [batch, in_features] -> [batch, in_features, 1]
        x = x.unsqueeze(-1)

        # grid: [in_features, num_grid_points]
        grid = self.grid

        # Initial basis: indicator functions (order 0)
        # bases[i,j,k] = 1 if grid[j,k] <= x[i,j] < grid[j,k+1], else 0
        bases = (
            (x >= grid[:, :-1]) & (x < grid[:, 1:])
        ).to(x.dtype)

        # Cox-de Boor recursion for higher order splines
        for k in range(1, self.spline_order + 1):
            # Left term: (x - t_i) / (t_{i+k} - t_i) * B_{i,k-1}(x)
            left_num = x - grid[:, :-(k + 1)]
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            # Add small epsilon to avoid division by zero in FP16
            left = left_num / (left_den + 1e-8) * bases[:, :, :-1]

            # Right term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:-k]
            right = right_num / (right_den + 1e-8) * bases[:, :, 1:]

            bases = left + right

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.

        Args:
            x: Input tensor of shape [batch_size, in_features] or
               [batch_size, seq_len, in_features]

        Returns:
            Output tensor of shape [batch_size, out_features] or
            [batch_size, seq_len, out_features]
        """
        # Handle 3D input (batch, seq, features)
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, self.in_features)

        # Ensure x is in valid range for splines (roughly)
        # This helps with numerical stability

        # 1. Base path: activation(x) @ base_weight
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 2. Spline path: spline(x) @ (spline_weight * spline_scaler)
        # Compute B-spline basis: [batch, in_features, num_coeffs]
        splines = self.compute_bspline_basis(x)

        # Scale spline weights: [out_features, in_features, num_coeffs]
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)

        # Reshape for linear operation
        # splines: [batch, in_features * num_coeffs]
        # scaled_weight: [out_features, in_features * num_coeffs]
        spline_output = F.linear(
            splines.view(x.size(0), -1),
            scaled_weight.view(self.out_features, -1),
        )

        # Combine outputs
        output = base_output + spline_output

        # Restore original shape if 3D
        if len(original_shape) == 3:
            output = output.reshape(batch_size, seq_len, self.out_features)

        return output


class EfficientKANProjector(nn.Module):
    """
    Two-layer KAN projector for mapping deepfake detection outputs to LLM space.

    Architecture: input_dim -> hidden_dim -> output_dim
    Default: 2 -> 128 -> 4096

    This replaces the MLP2x_GELU projector with a KAN-based alternative that
    uses learnable activation functions (B-splines) for potentially better
    expressiveness with fewer parameters.

    Args:
        input_dim: Input dimension (default: 2 for [P_real, P_fake])
        hidden_dim: Hidden layer dimension (default: 128)
        output_dim: Output dimension (default: 4096 for LLaMA hidden size)
        grid_size: Number of grid intervals for B-splines (default: 5)
        spline_order: Order of B-spline (default: 3 for cubic)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 4096,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Two-layer KAN
        self.layer1 = EfficientKANLinear(
            in_features=input_dim,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.layer2 = EfficientKANLinear(
            in_features=hidden_dim,
            out_features=output_dim,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through two-layer KAN.

        Args:
            x: Input tensor of shape [batch, 1, 2] or [batch, 2]

        Returns:
            Output tensor of shape [batch, 1, 4096] or [batch, 4096]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def get_param_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"EfficientKANProjector(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  grid_size={self.grid_size},\n"
            f"  spline_order={self.spline_order},\n"
            f"  params={self.get_param_count():,}\n"
            f")"
        )


# Convenience function for parameter counting
def count_kan_params(
    input_dim: int = 2,
    hidden_dim: int = 128,
    output_dim: int = 4096,
    grid_size: int = 5,
    spline_order: int = 3,
) -> int:
    """
    Calculate the number of parameters for a two-layer KAN.

    For each KAN layer:
    - base_weight: out * in
    - spline_weight: out * in * (grid + order)
    - spline_scaler: out * in

    Total per layer = out * in * (2 + grid + order)
    """
    num_coeffs = grid_size + spline_order

    # Layer 1: input_dim -> hidden_dim
    layer1_params = hidden_dim * input_dim * (2 + num_coeffs)

    # Layer 2: hidden_dim -> output_dim
    layer2_params = output_dim * hidden_dim * (2 + num_coeffs)

    return layer1_params + layer2_params


if __name__ == "__main__":
    # Test the implementation
    print("Testing EfficientKANProjector...")

    # Create projector
    projector = EfficientKANProjector(
        input_dim=2,
        hidden_dim=128,
        output_dim=4096,
        grid_size=5,
        spline_order=3,
    )
    print(projector)

    # Test forward pass with different input shapes
    # Shape 1: [batch, features]
    x1 = torch.randn(4, 2)
    y1 = projector(x1)
    print(f"Input: {x1.shape} -> Output: {y1.shape}")

    # Shape 2: [batch, seq, features] (as used in M2F2-Det)
    x2 = torch.randn(4, 1, 2)
    y2 = projector(x2)
    print(f"Input: {x2.shape} -> Output: {y2.shape}")

    # Test FP16 compatibility
    projector_fp16 = projector.half().cuda()
    x_fp16 = torch.randn(4, 1, 2, dtype=torch.float16, device='cuda')
    y_fp16 = projector_fp16(x_fp16)
    print(f"FP16 Input: {x_fp16.shape} -> Output: {y_fp16.shape}")

    # Test backward pass
    loss = y_fp16.sum()
    loss.backward()
    print("Backward pass successful!")

    # Compare with MLP parameter count
    mlp_params = 2 * 4096 + 4096 + 4096 * 4096 + 4096  # Linear(2, 4096) + Linear(4096, 4096)
    kan_params = projector.get_param_count()
    print(f"\nParameter comparison:")
    print(f"  MLP (2->4096->4096): {mlp_params:,}")
    print(f"  KAN (2->128->4096):  {kan_params:,}")
    print(f"  Reduction: {(1 - kan_params/mlp_params)*100:.1f}%")
