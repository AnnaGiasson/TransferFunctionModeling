from typing import Optional, Tuple

import numpy as np
from scipy import optimize, signal


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


class SystemModel:
    MAX_ZEROS: int = 20
    MAX_POLES: int = 20

    def __init__(self, max_zeros: int = 20, max_poles: int = 20) -> None:
        self.max_zeros = max_zeros
        self.max_poles = max_poles

        self.is_trained: bool = False

        # set after model is trained
        self.zeros: np.ndarray = None
        self.poles: np.ndarray = None
        self.gain: float = None

    def __str__(self) -> str:
        if not self.is_trained:
            return f"{self.__class__.__name__}()"

        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"zeros=({', '.join(map(str, self.zeros))}), ",
                f"poles=({', '.join(map(str, self.poles))}), ",
                f"gain={self.gain})",
            ]
        )

    def split_solver_coeffs(self, coeffs: np.ndarray) -> tuple[np.ndarray]:

        split_idx = self.max_zeros + 1

        return coeffs[0:split_idx], coeffs[split_idx:]  # (num, den)

    def error_function(
        self, coeffs: np.ndarray, f: np.ndarray, h: np.ndarray, reg: float
    ) -> float:

        num, den = self.split_solver_coeffs(coeffs)

        w = 2 * np.pi * np.array(f)
        _, h_predict = signal.freqs(num, den, worN=w)

        model_error = np.sum(np.abs(h_predict - h) ** 2)

        regularization_error = reg * np.linalg.norm(coeffs, ord=1)

        return model_error + regularization_error

    def fit(
        self, f: np.ndarray, h: np.ndarray, reg: Optional[float] = 0.0
    ) -> None:

        # set initial state of the model
        n_coeffs = self.max_zeros + self.max_poles + 2
        init_coeffs = 2 * np.random.random_sample(n_coeffs) - 1

        norm = Normalizer(f, h)

        result = optimize.minimize(
            fun=self.error_function,
            method="BFGS",
            x0=init_coeffs,
            args=(*norm.prescale(f, h), reg),
            tol=1e-1,
        )

        # check if optimization was successfull
        if not result.success:
            raise ArithmeticError("Model failed to converge", result.message)
        elif np.allclose(result.x, init_coeffs):
            raise ArithmeticError(
                "Error iterating beyond initial condition", result.message
            )

        # get polynomial coefficients for numerator and denominator.
        num, den = self.split_solver_coeffs(result.x)

        # make tf object
        self.tf = (
            norm.rescale_range(norm.rescale_domain(num)),
            norm.rescale_domain(den),
        )  # account for normalization on training data

        self.zeros, self.poles, self.gain = signal.tf2zpk(*self.tf)
        self.is_trained = True

    def predict(self, f: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise Exception(
                "Model must bee trained before prediction"
                "call self.fit first"
            )

        w = 2 * np.pi * f
        _, sig = signal.freqs(*self.tf, worN=w)
        return sig
