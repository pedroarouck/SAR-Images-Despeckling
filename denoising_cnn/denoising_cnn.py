# ─── 1) stdlib ──────────────────────────────────────────────────────────────────
import os
import random

# ─── 2) terceiros ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset

# ─── 3) seus módulos ───────────────────────────────────────────────────────────
from data       import SARPairDataset
from model      import DenoisingCNN
from loss       import CombinedLoss
from train      import train_epoch, validate_epoch
from checkpoint import save_checkpoint, load_checkpoint
from eval       import evaluate_and_plot, evaluate_metrics
from visualize  import plot_comparison

# ─── 4) hiperparâmetros e configuração ─────────────────────────────────────────
SEED            = 42
BATCH_SIZE      = 5
LR              = 1e-3
NUM_EPOCHS      = 1
NUM_WORKERS     = 4          # para DataLoader, se quiser parallelizar
DATA_ROOT       = "data/RESISC45"
L_LOOKS         = 5        # 1/5 = 0.2 de variância
FRACTION        = 0.001       # usar 1% das amostras para teste rápido

def run_experiment(
    name: str,
    LossFn,
    device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_ds: SARPairDataset,
    sample_idx: int
):
    """
    Executa um experimento de denoising:
      - name: rótulo ('baseline' ou 'stochastic')
      - LossFn: classe de loss (nn.MSELoss ou CombinedLoss)
    Retorna um dict com as métricas finais do modelo treinado.
    """
    print(f"\n>>> Iniciando experimento: {name}")

    # 1) modelo, loss e otimizador
    model     = DenoisingCNN(in_channels=1).to(device)
    criterion = LossFn() if name != "baseline" else nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scaler    = GradScaler() if device.type == "cuda" else None

    # 2) tentar carregar checkpoint
    ckpt_file = f"{name}_ckpt.pth"
    start_epoch = 0
    if os.path.exists(ckpt_file):
        print(f"Carregando checkpoint de '{ckpt_file}'…")
        start_epoch, cfg = load_checkpoint(ckpt_file, model, optimizer)
        print(f"Retomando da época {start_epoch+1}, lr={cfg['lr']}")

    # 3) loop de treino
    for epoch in range(start_epoch, NUM_EPOCHS):
        tr = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl = validate_epoch(model, val_loader,   criterion,              device)
        print(f"[{name}][{epoch+1}/{NUM_EPOCHS}] train={tr:.4f} val={vl:.4f}")
        save_checkpoint(model, optimizer, epoch, f"{name}_ckpt.pth")

    # 4) avaliação final (mesmo sample_idx em todos os experimentos)
    noisy_t, original_t = val_ds[sample_idx]
    noisy_b = noisy_t.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        denoised_b = model(noisy_b)

    original = original_t.squeeze().cpu().numpy()
    noisy    = noisy_t.squeeze().cpu().numpy()
    denoised = denoised_b.squeeze().cpu().numpy()

    metrics = evaluate_metrics(model, val_ds, device, sample_idx=sample_idx)
    return metrics, (original, noisy, denoised)


def main():
    # fixar seed
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Prepara dados com RESISC45 on-the-fly ---
    train_ds = SARPairDataset(
        root=DATA_ROOT,
        split="train",
        download=False,
        checksum=True,
        L=L_LOOKS
    )
    val_ds = SARPairDataset(
        root=DATA_ROOT,
        split="val",
        download=False,
        checksum=True,
        L=L_LOOKS
    )

    # ─── Limita o tamanho para testes rápidos ────────────────────────────
    num_train = int(len(train_ds) * FRACTION)
    num_val   = int(len(val_ds)   * FRACTION)
    train_ds = Subset(train_ds, list(range(num_train)))
    val_ds   = Subset(val_ds,   list(range(num_val)))  

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # escolhe um sample fixo para plotagem após treino
    sample_idx = random.randint(0, len(val_ds) - 1)

    experiments = [
        ("baseline", nn.MSELoss),
        ("stochastic", CombinedLoss),
    ]
    all_results = {}
    all_images  = {}

    for name, LossFn in experiments:
        res, imgs = run_experiment(
            name, LossFn, device,
            train_loader, val_loader, val_ds,
            sample_idx
        )
        all_results[name] = res
        all_images[name]  = imgs

    # exibe tabela de métricas
    print("\n=== Comparação de Métricas ===")
    header = "Métrica".ljust(25) + " | " + "Baseline".center(8) + " | " + "Stochastic".center(10)
    print(header)
    print("-" * len(header))
    for metric in all_results["baseline"].keys():
        b = all_results["baseline"][metric]
        s = all_results["stochastic"][metric]
        print(f"{metric.ljust(25)} | {b:8.4f} | {s:10.4f}")

    # plot final comparativo
    plot_comparison(all_images)


if __name__ == "__main__":
    main()
