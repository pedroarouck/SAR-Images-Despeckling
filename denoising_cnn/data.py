import torch
import torchvision.transforms.functional as F
from torchgeo.datasets import RESISC45

def add_speckle_noise(img: torch.Tensor, L: float) -> torch.Tensor:
    """
    Adiciona ruído speckle multiplicativo uniforme a um tensor de imagem.
    
    Parâmetros
    ----------
    img : torch.Tensor
        Tensor de imagem em escala de cinza, shape (1, H, W), valores em [0,1].
    L : float
        Número de looks; a variância do speckle será v = 1/L.
    
    Retorna
    -------
    torch.Tensor
        Tensor ruidoso, mesmo shape, valores em [0,1].
    """
    if L < 1:
        raise ValueError("O número de looks (L) deve ser >= 1.")
    v = 1.0 / L
    a = (3 * v) ** 0.5
    noise = torch.empty_like(img).uniform_(-a, a)
    noisy = (img + img * noise).clamp(0.0, 1.0)
    return noisy

class SARPairDataset(RESISC45):
    """
    Dataset que carrega RESISC45 via TorchGeo e devolve pares
    (noisy, clean), gerando speckle on-the-fly com a função add_speckle_noise.
    """

    def __init__(
        self,
        root: str,
        split: str,
        L: float = 8.0,
        download: bool = True,
        checksum: bool = True,
        transforms=None,
    ):
        super().__init__(
            root=root,
            split=split,
            download=download,
            checksum=checksum,
            transforms=transforms,
        )
        self.L = L

    def __getitem__(self, index):
        # 1) pega o dicionário retornado pelo RESISC45
        sample = super().__getitem__(index)
        img_uint8 = sample["image"]    # Tensor em [0,255], shape (C,H,W)
        # label = sample["label"]      # se você precisar do rótulo

        # 2) converter para float em [0,1]
        img = img_uint8.float().div(255.0)

        # 3) RGB → grayscale se necessário
        if img.shape[0] == 3:
            img = F.rgb_to_grayscale(img)  # shape (1,H,W)

        # 4) gerar o speckle via função separada
        clean = img
        noisy = add_speckle_noise(clean, self.L)

        return noisy, clean

    def __len__(self):
        return super().__len__()