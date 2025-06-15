import torch

def arith_geo_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Distância Aritmético-Geométrica entre distribuições p e q.

    Fórmula: D_AG(p, q) = Σ [ (p + q)/2 - sqrt(p * q) ]
    A soma é feita sobre todas as dimensões, retorna escalar médio por batch.

    Args:
        p: Tensor de probabilidades (batch, C, H, W)
        q: Tensor de probabilidades (mesma shape que p)
        eps: Pequeno valor para evitar raiz de zero ou divisão por zero

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

    # 4) Fórmula arithmético-geométrica
    ag = 0.5 * (p + q) - torch.sqrt(p * q)
    return ag.sum(dim=[1, 2, 3]).mean()