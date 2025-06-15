import matplotlib.pyplot as plt

def plot_results(original, noisy, denoised, figsize=(10,4)):
    """
    Plota três imagens lado a lado: original, ruidosa e denoised.
    `original`, `noisy`, `denoised` devem ser arrays 2D (H,W).
    """
    plt.figure(figsize=figsize)
    titles = ['Original', 'Noisy', 'Denoised']
    for i, img in enumerate([original, noisy, denoised], 1):
        plt.subplot(1, 3, i)
        plt.title(titles[i-1])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
