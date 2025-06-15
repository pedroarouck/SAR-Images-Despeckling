import torch

def triangular_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Distância Triangular entre distribuições p e q.

    Fórmula: D_T(p, q) = Σ [ (p - q)² / (p + q) ]

    Args:
        p: Tensor de probabilidades (batch, C, H, W)
        q: Tensor de probabilidades (mesma shape que p)
        eps: Pequeno valor para evitar divisão por zero

    Returns:
        Tensor escalar com valor médio da distância no batch
    """
    # 1) Garantir não-negatividade
    p = p.clamp(min=0)
    q = q.clamp(min=0)

    # 2) Normalizar cada batch para soma=1
    sum_p = p.sum(dim=[1,2,3], keepdim=True)
    sum_q = q.sum(dim=[1,2,3], keepdim=True)
    p = p / (sum_p + eps)
    q = q / (sum_q + eps)

    # 3) Evitar zeros para o log
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    # 4) Fórmula da distância triangular
    numerator = (p - q) ** 2
    denominator = p + q
    tri = numerator / (denominator + eps)
    return tri.sum(dim=[1, 2, 3]).mean()
