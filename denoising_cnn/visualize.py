import matplotlib.pyplot as plt

def plot_results(original, noisy, denoised, title_prefix="", figsize=(10,4)):
    """
    Plota três imagens lado a lado: original, ruidosa e denoised.
    `original`, `noisy`, `denoised` devem ser arrays 2D (H,W).
    """
    plt.figure(figsize=figsize)
    titles = [f"{title_prefix} — Original", 
            f"{title_prefix} — Noisy", 
            f"{title_prefix} — Denoised"]
    for i, img in enumerate([original, noisy, denoised], 1):
        plt.subplot(1, 3, i)
        plt.title(titles[i-1])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_comparison(images_dict, figsize=(12, 8)):
    """
    Mostra, em duas linhas, as tríades (Original, Noisy, Denoised)
    para cada método (baseline, stochastic), incluindo o nome do modo
    em cada título de subplot.
    """
    modes = list(images_dict.keys())  # ex: ['baseline','stochastic']
    labels = ['Original', 'Noisy', 'Denoised']
    fig, axes = plt.subplots(len(modes), 3, figsize=figsize)

    for i, mode in enumerate(modes):
        orig, noisy, denoised = images_dict[mode]
        for j, img in enumerate([orig, noisy, denoised]):
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            # título explicando método + etapa
            ax.set_title(f"{mode.capitalize()} — {labels[j]}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()