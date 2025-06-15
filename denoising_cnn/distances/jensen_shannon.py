import torch
import torch.nn.functional as F

def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Divergência de Jensen-Shannon entre distribuições p e q.

    Fórmula: JS(p || q) = 0.5 * KL(p || m)  0.5 * KL(q || m), onde m = (p  q) / 2

    Args:
        p: Tensor de probabilidades (batch, C, H, W)
        q: Tensor de probabilidades (mesma shape que p)
        eps: Pequeno valor para evitar log de zero

    Returns:
        Tensor escalar com valor médio da divergência no batch
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

    # 4) Fórmula JS
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log(p / m)).sum(dim=[1,2,3])
    kl_qm = (q * torch.log(q / m)).sum(dim=[1,2,3])
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js.mean()
