import os
import torch

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Salva o estado atual do modelo, optimizer e epoch em um arquivo.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr'],
    }
    torch.save(state, filepath)
    print(f"Checkpoint salvo em: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """
    Carrega um checkpoint salvo; retorna epoch e config (e opcionalmente carrega optimizer).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo {filepath} não foi encontrado.")
    state = torch.load(filepath)
    model.load_state_dict(state['model_state_dict'])
    print(f"Modelo carregado de: {filepath}")
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print("Estado do otimizador restaurado.")
    return state['epoch'], {'lr': state.get('lr', None)}
