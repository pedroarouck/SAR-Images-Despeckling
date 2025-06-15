import torch

def renyi_divergence(p: torch.Tensor, q: torch.Tensor, alpha: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """
    Divergência de Rényi entre distribuições p e q.

    Fórmula: D_α(p || q) = (1 / (α - 1)) * log(Σ [pᵅ * q¹⁻ᵅ])

    Args:
        p: Tensor de probabilidades (batch, C, H, W)
        q: Tensor de probabilidades (mesma shape que p)
        alpha: Parâmetro de ordem da divergência (α ≠ 1)
        eps: Pequeno valor para evitar divisão por zero

    Returns:
        Tensor escalar com valor médio da divergência no batch
    """
    if alpha == 1.0:
        raise ValueError("Para α=1 use a divergência de Kullback-Leibler.")

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

    # 4) Fórmula da divergência de Rényi

    pq_alpha = (p.pow(alpha) * q.pow(1 - alpha)).sum(dim=[1, 2, 3])
    divergence = (1.0 / (alpha - 1)) * torch.log(pq_alpha + eps)
    return divergence.mean()
