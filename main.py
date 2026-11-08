"""Demonstrate all figures.py outputs with synthetic data: python main.py."""

from pathlib import Path

import numpy as np

from mypylibs import figures


class DemoDescriptors:
    """Small sample-data adapter for the interface used by save_final."""

    def __init__(self):
        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        frequencies = np.array([1] + [k for n in range(2, 12) for k in (n, -n)])
        amplitudes = 0.35 / np.arange(1, len(frequencies) + 1) ** 1.4
        amplitudes[0] = 1
        self.components = amplitudes[:, None] * np.exp(1j * frequencies[:, None] * t)
        self.z = self.components.sum(axis=0)

    def num_harmonics(self, percent):
        return max(1, int(np.ceil(len(self.components) * percent / 100)))

    def reconstruct(self, count):
        return self.components[:count].sum(axis=0)


def main():
    output = Path(__file__).resolve().parent / 'out'
    output.mkdir(exist_ok=True)

    # Sample panels cover RGB, grayscale, signed values, and boolean masks.
    y, x = np.mgrid[-1:1:128j, -1:1:128j]
    gray = np.exp(-4 * (x ** 2 + y ** 2))
    rgb = np.stack((gray, 0.6 * gray, 1 - gray), axis=-1)
    steps = [
        ('Synthetic RGB image', rgb),
        ('Synthetic grayscale image', gray),
        ('Synthetic signed response', x * gray),
        ('Synthetic threshold mask', gray > 0.4),
    ]
    descriptors = DemoDescriptors()

    figures.save_transforms(steps, output / 'demo_transforms.png')
    figures.save_trajectory(descriptors.z, output / 'demo_trajectory.png')
    figures.save_final(descriptors, output / 'demo_final.png')
    print(f'Saved demo_transforms.png, demo_trajectory.png, and demo_final.png to {output}')


if __name__ == '__main__':
    main()
