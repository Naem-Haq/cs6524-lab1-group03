
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK = '#1f4e79'
GHOST = '#d5dae0'
MARKER = '#c8442b'
PERCENTS = range(5, 105, 5)


def _frame_curve(ax, z):
    margin = .06 * max(np.ptp(z.real), np.ptp(z.imag))
    ax.set_xlim(z.real.min() - margin, z.real.max() + margin)
    ax.set_ylim(z.imag.max() + margin, z.imag.min() - margin)
    ax.set_aspect('equal')
    ax.axis('off')


def _save(fig, path):
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_transforms(steps, path):
    cols = 4
    rows = int(np.ceil(len(steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows), layout='constrained')
    for ax in axes.flat:
        ax.axis('off')
    for ax, (title, image) in zip(axes.flat, steps):
        if image.dtype == bool:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        elif image.ndim == 2 and image.min() < 0:
            limit = np.abs(image).max()
            ax.imshow(image, cmap='RdBu_r', vmin=-limit, vmax=limit)
        elif image.ndim == 2:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=10)
    _save(fig, path)


def save_trajectory(z, path):
    closed = np.append(z, z[0])
    T = len(z)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), layout='constrained')
    for ax, t in zip(axes.flat, np.linspace(0, T, 10).round().astype(int)):
        ax.plot(closed.real, closed.imag, color=GHOST, lw=1)
        ax.plot(closed.real[:t + 1], closed.imag[:t + 1], color=INK, lw=1.6)
        ax.plot(closed.real[t], closed.imag[t], 'o', color=MARKER, ms=5)
        ax.set_title(f't = {t} / {T}', fontsize=10)
        _frame_curve(ax, z)
    _save(fig, path)


def save_final(descriptors, path):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), layout='constrained')
    for ax, percent in zip(axes.flat, PERCENTS):
        K = descriptors.num_harmonics(percent)
        curve = descriptors.reconstruct(K)
        ax.plot(np.append(curve.real, curve.real[0]), np.append(curve.imag, curve.imag[0]),
                color=INK, lw=1.2)
        ax.set_title(f'{percent}% (K = {K})', fontsize=10)
        _frame_curve(ax, descriptors.z)
    _save(fig, path)
