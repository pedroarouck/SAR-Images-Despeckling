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

def evaluate_metrics(model, val_dataset, device, sample_idx=None, renyi_alpha=0.7):
    """
    Retorna um dict com as métricas para uma amostra aleatória de val_dataset:
      'KL', 'Rényi', 'Hellinger', 'Bhattacharyya', 'Jensen–Shannon',
      'Aritmético–Geométrica', 'Triangular', 'Média Harmônica',
      'PSNR', 'SSIM'
    """
    # 1) Seleciona um sample
    if sample_idx is None:
        idx = np.random.randint(len(val_dataset))
    else:
        idx = sample_idx
    noisy_t, original_t = val_dataset[idx]
    noisy_b  = noisy_t.unsqueeze(0).to(device)

    # 2) Inferência
    model.eval()
    with torch.no_grad():
        denoised_b = model(noisy_b)

    # 3) Converte para numpy 2D
    denoised = denoised_b.squeeze().cpu().numpy()
    original = original_t.squeeze().cpu().numpy()
    noisy    = noisy_t.squeeze().cpu().numpy()

    # 4) Tensores para distâncias
    to_tensor = lambda x: torch.from_numpy(x)[None,None].to(device)
    t_o = to_tensor(original)
    t_d = to_tensor(denoised)

    # 5) Calcula métricas estocásticas
    results = {
        'KL':     kl_divergence(t_o, t_d).item(),
        'Rényi':  renyi_divergence(t_o, t_d, alpha=renyi_alpha).item(),
        'Hellinger':             hellinger_distance(t_o, t_d).item(),
        'Bhattacharyya':         bhattacharyya_distance(t_o, t_d).item(),
        'Jensen–Shannon':        jensen_shannon_divergence(t_o, t_d).item(),
        'Aritmético–Geométrica': arith_geo_distance(t_o, t_d).item(),
        'Triangular':            triangular_distance(t_o, t_d).item(),
        'Média Harmônica':       harmonic_mean_distance(t_o, t_d).item(),
    }

    # 6) PSNR e SSIM
    results['PSNR'] = psnr(original, denoised, data_range=1.0)
    results['SSIM'] = ssim(original, denoised, data_range=1.0)

    return results

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
