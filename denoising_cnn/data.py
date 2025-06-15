import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_images_from_folder(folder):
    """
    Carrega imagens em escala de cinza de uma pasta e valida erros durante o processo.
    """
    images = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"A pasta '{folder}' não existe.")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"O caminho '{folder}' não é uma pasta válida.")
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            continue
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
    if not images:
        raise ValueError("Nenhuma imagem válida foi carregada.")
    return images

def add_speckle_noise(image, L=1):
    """
    Adiciona ruído speckle à imagem usando a distribuição Gamma.
    """
    if L < 1:
        raise ValueError("O número de looks (L) deve ser >=1.")
    row, col = image.shape
    gamma_noise = np.random.gamma(L, 1.0 / L, (row, col))
    return image * gamma_noise

class ImageDataset(Dataset):
    """
    Dataset de pares (ruidosa, limpa), opcionalmente com augmentations.
    """
    def __init__(self, noisy_images, clean_images, apply_augmentations=False):
        if len(noisy_images) != len(clean_images):
            raise ValueError("Tamanhos diferentes em noisy_images e clean_images.")
        self.noisy_images = noisy_images
        self.clean_images = clean_images
        self.transform = None
        if apply_augmentations:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
            ])

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy = self.noisy_images[idx]
        clean = self.clean_images[idx]
        if self.transform:
            noisy = self._apply_transform(noisy)
            clean = self._apply_transform(clean)
        # Converter para tensores com shape (1, H, W)
        noisy_arr = np.asarray(noisy)
        clean_arr = np.asarray(clean)

        # Se veio com shape (H, W, 1), retire o último eixo
        if noisy_arr.ndim == 3 and noisy_arr.shape[2] == 1:
            noisy_arr = noisy_arr[:, :, 0]
        if clean_arr.ndim == 3 and clean_arr.shape[2] == 1:
             clean_arr = clean_arr[:, :, 0]
 
        # Agora (H, W) → tensor (1, H, W)
        noisy_t = torch.from_numpy(noisy_arr).float().unsqueeze(0)
        clean_t = torch.from_numpy(clean_arr).float().unsqueeze(0)
        return noisy_t, clean_t


    def _apply_transform(self, image):
        image_pil = transforms.ToPILImage()(image)
        aug_t = self.transform(image_pil)
        # transforms.ToTensor() → tensor shape (C, H, W); em grayscale C=1
        arr = transforms.ToTensor()(aug_t)
        # volta para numpy (H, W)
        return arr.squeeze(0).numpy()