import torch

def hellinger_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Distância de Hellinger entre distribuições p e q.

    Fórmula: H(p, q) = (1 / √2) * ||√p - √q||₂

    Args:
        p: Tensor de probabilidades (batch, C, H, W)
        q: Tensor de probabilidades (mesma shape que p)
        eps: Pequeno valor para evitar raiz de zero

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

    # 4) Fórmula da distância de Hellinger
    p_sqrt = torch.sqrt(p.clamp(min=eps))
    q_sqrt = torch.sqrt(q.clamp(min=eps))
    diff = p_sqrt - q_sqrt
    norm = torch.norm(diff, p=2, dim=[1, 2, 3])
    return (norm / torch.sqrt(torch.tensor(2.0))).mean()
