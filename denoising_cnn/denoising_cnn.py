# ─── 1) stdlib ──────────────────────────────────────────────────────────────────
import os
import random

# ─── 2) terceiros ──────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ─── 3) seus módulos ───────────────────────────────────────────────────────────
from data       import load_images_from_folder, add_speckle_noise, ImageDataset
from model      import DenoisingCNN
from loss       import CombinedLoss
from train      import train_epoch, validate_epoch
from checkpoint import save_checkpoint, load_checkpoint
from eval       import evaluate_and_plot
from eval import evaluate_metrics
from visualize  import plot_comparison

# ─── 4) hiperparâmetros e configuração ─────────────────────────────────────────
SEED            = 42
DATA_DIR        = 'BSDS500-master/BSDS500/data/images/train'
IMG_SIZE        = (128, 128)
FRACTION        = 0.1        # usar 10% das imagens
BATCH_SIZE      = 4
LR              = 1e-3
NUM_EPOCHS      = 10
NUM_WORKERS     = 4          # para DataLoader, se quiser parallelizar

def run_experiment(name: str, LossFn, device, train_loader, val_loader, val_ds, sample_idx):
    """
    Executa um experimento de denoising:
      - name: rótulo ('baseline' ou 'stochastic')
      - LossFn: classe de loss (nn.MSELoss ou CombinedLoss)
    Retorna um dict com as métricas finais do modelo treinado.
    """
    print(f"\n>>> Iniciando experimento: {name}")

    # --- Escolhe UM índice aleatório para TODOS os experimentos ---
    # 2) Modelo, loss e otimizador
    model     = DenoisingCNN(in_channels=1).to(device)
    if name == "baseline":
        criterion = nn.MSELoss()
    else:
        criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scaler    = GradScaler() if device.type == "cuda" else None

    # 2.1) Tentar carregar checkpoint
    ckpt_file = f"{name}_ckpt.pth"
    start_epoch = 0
    if os.path.exists(ckpt_file):
        print(f"Carregando checkpoint de '{ckpt_file}'…")
        # só model precisa, mas passamos optimizer para restaurar state_dict
        start_epoch, cfg = load_checkpoint(ckpt_file, model, optimizer)
        print(f"Retomando da época {start_epoch+1}, lr={cfg['lr']}")
   
    # 3) Treino
    for epoch in range(start_epoch, NUM_EPOCHS):
        tr = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl = validate_epoch(model, val_loader,   criterion,              device)
        print(f"[{name}][{epoch+1}/{NUM_EPOCHS}] train={tr:.4f} val={vl:.4f}")
        save_checkpoint(model, optimizer, epoch, f"{name}_ckpt.pth")

       # 4) Avaliação final usando sample_idx fixo
    # extrai as mesmas imagens de val_ds
    noisy_t, original_t = val_ds[sample_idx]
    noisy_b = noisy_t.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        denoised_b = model(noisy_b)

    original = original_t.squeeze().cpu().numpy()
    noisy    = noisy_t.squeeze().cpu().numpy()
    denoised = denoised_b.squeeze().cpu().numpy()

    metrics = evaluate_metrics(model, val_ds, device, sample_idx=sample_idx)
    # (você precisará adicionar esse parâmetro em evaluate_metrics também,
    # mas se não quiser mexer lá, basta chamar a evaluate_metrics original
    # para fins de métrica, e só usar sample_idx aqui para capturar imagens.)

    return metrics, (original, noisy, denoised)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- Prepara dados UMA ÚNICA VEZ ---
    images = load_images_from_folder(DATA_DIR)
    images = [cv2.resize(img, IMG_SIZE) for img in images]
    images = images[: int(len(images) * FRACTION)]
    noisy  = [add_speckle_noise(img, L=5) for img in images]

    images = np.array(images) / 255.0
    noisy  = np.array(noisy)  / 255.0
    images = images[..., None]
    noisy  = noisy[...,  None]

    X_train, X_val, y_train, y_val = train_test_split(
        noisy, images, test_size=0.2, random_state=SEED
    )
    train_ds = ImageDataset(X_train, y_train)
    val_ds   = ImageDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    sample_idx = random.randint(0, len(val_ds) - 1)
    # --------------------------------------
    experiments = [
        ("baseline", nn.MSELoss),
        ("stochastic", CombinedLoss),
    ]
    all_results = {}
    all_images  = {}

    for name, LossFn in experiments:
        res, imgs = run_experiment(name, LossFn, device,
                                train_loader, val_loader, val_ds,
                                sample_idx)
        all_results[name] = res
        all_images[name]  = imgs

    # Avaliação e plotagem final
    print("\n=== Comparação de Métricas ===")
    header = "Métrica".ljust(25) + " | " + "Baseline".center(8) + " | " + "Stochastic".center(10)
    print(header)
    print("-" * len(header))
    for metric in all_results["baseline"].keys():
        b = all_results["baseline"][metric]
        s = all_results["stochastic"][metric]
        print(f"{metric.ljust(25)} | {b:8.4f} | {s:10.4f}")

    # 3) Plot comparativo final
    plot_comparison(all_images)

if __name__ == "__main__":
    main()
