
import numpy as np


class Descriptors:
    """DFT of z[n] = x[n] + iy[n], with harmonics ordered from most to least dominant."""

    def __init__(self, x, y):
        self.z = x + 1j * y
        self.N = len(self.z)
        spectrum = np.fft.fft(self.z)
        frequencies = np.fft.fftfreq(self.N)

        # The DC term only positions the shape, so it leads regardless of amplitude
        rest = 1 + np.argsort(-np.abs(spectrum[1:]))
        self.order = np.concatenate(([0], rest))
        self.F = spectrum[self.order]
        self.omega = frequencies[self.order]

    def num_harmonics(self, percent):
        return max(1, round(self.N * percent / 100))

    def reconstruct(self, K):
        truncated = np.zeros(self.N, dtype=complex)
        truncated[self.order[:K]] = self.F[:K]
        return np.fft.ifft(truncated)

    def arm(self, n):
        """Tip-to-tail positions of all rotating vectors at (possibly fractional) sample n."""
        return np.cumsum(self.F / self.N * np.exp(2j * np.pi * self.omega * n))
