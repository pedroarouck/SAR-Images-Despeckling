# ─── 1) stdlib ──────────────────────────────────────────────────────────────────
import os
import random

# ─── 2) terceiros ──────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
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

# ─── 4) hiperparâmetros e configuração ─────────────────────────────────────────
SEED            = 42
DATA_DIR        = 'BSDS500-master/BSDS500/data/images/train'
IMG_SIZE        = (128, 128)
FRACTION        = 0.1        # usar 10% das imagens
BATCH_SIZE      = 4
LR              = 1e-3
NUM_EPOCHS      = 10
CHECKPOINT_PATH = 'checkpoint.pth'
NUM_WORKERS     = 4          # para DataLoader, se quiser parallelizar

def main():
    # 1) fixar seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 2) carregar e preparar dados
    images = load_images_from_folder(DATA_DIR)
    images = [cv2.resize(img, IMG_SIZE) for img in images]
    images = images[: int(len(images) * FRACTION)]
    noisy  = [add_speckle_noise(img, L=5) for img in images]

    images        = np.array(images) / 255.0
    noisy         = np.array(noisy)  / 255.0
    images       = images[..., np.newaxis]
    noisy        = noisy[...,  np.newaxis]

    X_train, X_val, y_train, y_val = train_test_split(
        noisy, images, test_size=0.2, random_state=SEED
    )
    train_ds = ImageDataset(X_train, y_train)
    val_ds   = ImageDataset(X_val,   y_val)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 3) configurar modelo, loss, otimizador e scaler
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = DenoisingCNN(in_channels=1).to(device)
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    use_amp = (device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    # 4) carregar checkpoint se existir
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch, cfg = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer
        )
        print(f"Retomando da época {start_epoch+1} — lr={cfg['lr']}")

    # 5) loop de treinamento
    for epoch in range(start_epoch, NUM_EPOCHS):
        tr_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        print(f"[{epoch+1}/{NUM_EPOCHS}] train={tr_loss:.4f} val={val_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)

    # 6) avaliação e plotagem final
    evaluate_and_plot(model, val_ds, device)


if __name__ == "__main__":
    main()
