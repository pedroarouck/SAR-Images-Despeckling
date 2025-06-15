import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """
    Executa uma época de treinamento.
    Retorna a perda média (float).
    """
    model.train()
    running_loss = 0.0
    use_amp = (device.type == "cuda")
    for noisy, clean in tqdm(train_loader, desc='Training', leave=False):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        # 1) Forward com AMP apenas se for GPU
        with autocast(enabled=use_amp):
            outputs = model(noisy)
            loss    = criterion(outputs, clean)

        # 2) Backward: use GradScaler se for GPU, senão normal
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * noisy.size(0)
    return running_loss / len(train_loader.dataset)

def validate_epoch(model, val_loader, criterion, device):
    """
    Executa uma época de validação (sem gradientes).
    Retorna a perda média (float).
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for noisy, clean in tqdm(val_loader, desc='Validation', leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            loss    = criterion(outputs, clean)
            running_loss += loss.item() * noisy.size(0)
    return running_loss / len(val_loader.dataset)
