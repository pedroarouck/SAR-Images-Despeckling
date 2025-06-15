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
        images_dict deve ser:
    {'baseline': (orig_b, noisy_b, den_b),
    'stochastic': (orig_s, noisy_s, den_s)}
    Cada imagem é array 2D.
    """
    modes = list(images_dict.keys())  # ['baseline','stochastic']
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for i, mode in enumerate(modes):
        orig, noisy, denoised = images_dict[mode]
        for j, img in enumerate([orig, noisy, denoised]):
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            if i == 0:
                ax.set_title(['Original', 'Noisy', 'Denoised'][j])
            ax.set_ylabel(mode if j == 0 else "")
            ax.axis('off')
    plt.tight_layout()
    plt.show()
