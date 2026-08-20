# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lloyd–Max scalar codebook for TurboQuantMSEV2 (ported from turboquant-npu).

After rotating a d-dimensional unit vector by a random orthogonal matrix, each
coordinate is approximated by N(0, 1/d) for large d. Centroids solve the
Lloyd–Max conditions via SciPy integration.

Reference implementation: ``turboquant-npu/lloyd_max.py`` (TurboQuant paper).
"""

from __future__ import annotations

import math

import torch
from scipy import integrate


def beta_pdf(x: float, d: int) -> float:
    """PDF of a single coordinate after random rotation of a d-dim unit vector."""
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1 - x * x) ** ((d - 3) / 2)


def gaussian_approx_pdf(x: float, d: int) -> float:
    """Gaussian approximation N(0, 1/d) — accurate for d >= 64."""
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def solve_lloyd_max(
    d: int,
    bits: int,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd–Max quantizer for the coordinate distribution."""
    n_levels = 2**bits
    pdf = (lambda x: beta_pdf(x, d)) if use_exact else (lambda x: gaussian_approx_pdf(x, d))
    sigma = 1.0 / math.sqrt(d)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _iteration in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids: list[float] = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]

            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)

            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids

        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


class LloydMaxCodebook:
    """Precomputed Lloyd–Max codebook for a given dimension and bit-width."""

    def __init__(self, d: int, bits: int, use_exact: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 2**bits
        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        diffs = x.unsqueeze(-1) - self.centroids.to(x.device)
        return diffs.abs().argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(indices.device)[indices]

    def __repr__(self) -> str:
        return (
            f"LloydMaxCodebook(d={self.d}, bits={self.bits}, levels={self.n_levels})"
        )
