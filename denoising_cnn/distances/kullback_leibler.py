import torch

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Divergência de Kullback-Leibler entre distribuições p e q.

    Fórmula: KL(p || q) = Σ [ p * log(p / q) ], assumindo p, q >= 0 e normalizados por canal.

    Args:
        p: Tensor de probabilidades (batch, C, H, W)
        q: Tensor de probabilidades (mesma shape que p)
        eps: Pequeno valor para evitar divisão por zero e log(0)

    Returns:
        Tensor escalar com valor médio da divergência no batch
    """
   # 1) Garantir não-negatividade
    p = p.clamp(min=0)
    q = q.clamp(min=0)

    # 2) Normalizar cada tensor para soma = 1
    sum_p = p.sum(dim=[1,2,3], keepdim=True)
    sum_q = q.sum(dim=[1,2,3], keepdim=True)
    p = p / (sum_p + eps)
    q = q / (sum_q + eps)

    # 3) Evitar log(0)
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    # 4) Cálculo da divergência
    kl_map = p * torch.log(p / q)
    return kl_map.sum(dim=[1,2,3]).mean()
