from typing import Tuple

import numpy as np


class Normalizer:
    """
    Normalizes frequency response data such that the domain of frequencies,
    f, always span the range [0, 1] and the magnitude of the (complex) data,
    h, is within the range (0, 1]
    """

    def __init__(self, f: np.ndarray, h: np.ndarray) -> None:

        self.f_scale_factor = 1 / np.max(f)

        # h is assumed to be an array of complex numbers
        self.h_scale_factor = 1 / np.max(np.abs(h))

    def prescale(
        self, f: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """normilize data before being used to fit a model"""
        return f * self.f_scale_factor, h * self.h_scale_factor

    def rescale_domain(self, coeffs: np.ndarray) -> np.ndarray:
        """
        rescales the polynomial coefficients of a transfer function (numerator
        or denominator) to compensate for the prescaling done on the domain of
        the training data
        """
        N = len(coeffs)
        scale_factors = np.vander(self.f_scale_factor * np.ones(N))[0, :]
        return scale_factors * coeffs

    def rescale_range(self, coeffs: np.ndarray) -> np.ndarray:
        """
        rescales the numerator polynomial coefficients of a transfer function
        to compensate for the prescaling done on the range of the training data
        """
        return coeffs / self.h_scale_factor


def magnitude(x: np.ndarray) -> np.ndarray:
    """returns the magnitude of a complex vector"""
    return np.abs(x)


def magnitude_db(x: np.ndarray) -> np.ndarray:
    """returns the magnitude of a complex vector in units of dB"""
    return 20 * np.log(magnitude(x))


def phase(x: np.ndarray) -> np.ndarray:
    """returns the phase of a complex vector in radians"""
    return np.angle(x)


def phase_deg(x: np.ndarray) -> np.ndarray:
    """returns the phase of a complex vector in units of degrees"""
    return phase(x) * (180 / np.pi)
