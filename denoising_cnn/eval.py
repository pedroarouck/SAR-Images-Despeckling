# eval.py

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
from visualize import plot_results

def evaluate_and_plot(model, val_dataset, device, renyi_alpha=0.7):
    """
    Seleciona um sample aleatório de val_dataset, roda o modelo, calcula todas as métricas
    e faz o plot com plot_results.

    Args:
        model: instância de DenoisingCNN já carregada e em modo eval.
        val_dataset: instância de ImageDataset (sem DataLoader).
        device: 'cpu' ou 'cuda'.
        renyi_alpha: valor de alpha para a divergência de Rényi.
    """
    # 1) Escolhe um índice aleatório
    idx = np.random.randint(len(val_dataset))
    noisy_t, original_t = val_dataset[idx]  
    # noisy_t: tensor (1,H,W), original_t: tensor (1,H,W)

    # 2) Para rodar no modelo, adiciona batch dim
    noisy_b = noisy_t.unsqueeze(0).to(device)   # shape (1,1,H,W)

    # 3) Roda o modelo
    with torch.no_grad():
        denoised_b = model(noisy_b)
    # denommit batch & canal
    denoised = denoised_b.squeeze().cpu().numpy()   # (H,W)
    original = original_t.squeeze().cpu().numpy()   # (H,W)
    noisy = noisy_t.squeeze().cpu().numpy()         # (H,W)

    # 4) Prepara tensores para distâncias (batch,channel,H,W)
    t_orig = torch.from_numpy(original)[None,None].to(device)
    t_den  = torch.from_numpy(denoised)[None,None].to(device)

    # 5) Calcula métricas
    metrics = {
        'KL':     kl_divergence(t_orig, t_den).item(),
        'Rényi':  renyi_divergence(t_orig, t_den, alpha=renyi_alpha).item(),
        'Hellinger':             hellinger_distance(t_orig, t_den).item(),
        'Bhattacharyya':         bhattacharyya_distance(t_orig, t_den).item(),
        'Jensen–Shannon':        jensen_shannon_divergence(t_orig, t_den).item(),
        'Aritmético–Geométrica': arith_geo_distance(t_orig, t_den).item(),
        'Triangular':            triangular_distance(t_orig, t_den).item(),
        'Média Harmônica':       harmonic_mean_distance(t_orig, t_den).item(),
        'PSNR':  psnr(original, denoised, data_range=1.0),
        'SSIM':  ssim(original, denoised, data_range=1.0),
    }

    # 6) Exibe resultados
    print("\n==== Avaliação Final ====")
    for name, val in metrics.items():
        print(f"{name:20s}: {val:.6f}")

    # 7) Plota
    plot_results(original, noisy, denoised)
