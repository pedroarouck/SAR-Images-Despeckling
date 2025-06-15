import torch
import torch.nn as nn
import torch.nn.functional as F

from distances import (
    kl_divergence,
    renyi_divergence,
    hellinger_distance,
    bhattacharyya_distance,
    jensen_shannon_divergence,
    arith_geo_distance,
    triangular_distance,
    harmonic_mean_distance,
)

class CombinedLoss(nn.Module):
    """
    Perda composta que combina MSE + diversas métricas de divergência/distância.
    """

    def __init__(self):
        super().__init__()
        # Pesos aprendíveis para algumas métricas
        self.alpha_kl   = nn.Parameter(torch.tensor(0.5))
        self.beta_js   = nn.Parameter(torch.tensor(0.3))
        self.gamma_hel = nn.Parameter(torch.tensor(0.2))
        # (Opcional) adicionar outros pesos aqui

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(output, target)

        kl  = kl_divergence(output, target)
        js  = jensen_shannon_divergence(output, target)
        hel = hellinger_distance(output, target)

        # As demais métricas, sem peso adicional por padrão
        ag  = arith_geo_distance(output, target)
        bh  = bhattacharyya_distance(output, target)
        hm  = harmonic_mean_distance(output, target)
        tri = triangular_distance(output, target)
        rd  = renyi_divergence(output, target, alpha=0.7)

        total = (
            mse
            + self.alpha_kl  * kl
            + self.beta_js   * js
            + self.gamma_hel * hel
            + ag + bh + hm + tri + rd
        )
        return total
