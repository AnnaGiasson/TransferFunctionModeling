from dataclasses import dataclass
from numbers import Number
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np
from scipy import optimize, signal

from plotting_common import Series, multiseries_bodeplot


class System(Protocol):
    def __str__(self) -> str:
        """
        returns a string with the name of the class in addition to its transfer
        function, in the form of its zeros, poles, and gain
        """

    def measure(self, f: np.ndarray) -> np.ndarray:
        """Samples the frequency response of the system at frequencies "f" """


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


# generate example systems
@dataclass
class KnownSystem:
    zeros: Iterable[Number]
    poles: Iterable[Number]
    gain: float

    def __post_init__(self) -> None:
        self.tf = signal.zpk2tf(self.zeros, self.poles, self.gain)

    def __str__(self) -> str:
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"zeros=({', '.join(map(str, self.zeros))}), ",
                f"poles=({', '.join(map(str, self.poles))}), ",
                f"gain={self.gain})",
            ]
        )

    def measure(self, f: np.ndarray) -> np.ndarray:
        w = 2 * np.pi * np.array(f)
        return signal.freqs(*self.tf, worN=w)[1]


class RandomSystem:

    MAX_ZEROS: int = 20
    MAX_POLES: int = 20
    EIGENVALUE_RANGE: Tuple[float] = (-10e6, 10e6)
    GAIN_RANGE: Tuple[float] = (1e-3, 1e3)

    def __init__(
        self, n_zeros: Optional[int] = None, n_poles: Optional[int] = None
    ) -> None:

        if n_zeros is None:
            n_zeros = np.random.randint(0, self.MAX_ZEROS + 1)
        if n_poles is None:
            n_poles = np.random.randint(0, self.MAX_POLES + 1)
        self.n_zeros = n_zeros
        self.n_poles = n_poles

        self.zeros = self.select_eigenvalues(self.n_zeros)
        self.poles = self.select_eigenvalues(self.n_poles)
        self.gain = self.uniform_sample(*self.GAIN_RANGE)
        self.tf = signal.zpk2tf(self.zeros, self.poles, self.gain)

    def __str__(self) -> str:
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"zeros=({', '.join(map(str, self.zeros))}), ",
                f"poles=({', '.join(map(str, self.poles))}), ",
                f"gain={self.gain})",
            ]
        )

    @staticmethod
    def uniform_sample(val_min: Number, val_max: Number, n: int = 1) -> Number:
        if n == 1:
            return (val_max - val_min) * np.random.random_sample() + val_min
        return (val_max - val_min) * np.random.random_sample(n) + val_min

    def select_eigenvalues(
        self, n_vals: int, p_complex: float = 0.2
    ) -> List[Number]:

        eigen_vals: List[Number] = []

        while len(eigen_vals) < n_vals:
            is_complex = np.random.choice(
                [True, False], p=[p_complex, 1 - p_complex]
            )

            if is_complex:  # complex conjugates

                # need to be in pairs to be real systems
                if (n_vals - len(eigen_vals)) < 2:
                    continue
                real = self.uniform_sample(*self.EIGENVALUE_RANGE)
                imag = self.uniform_sample(*self.EIGENVALUE_RANGE)

                eigen_vals.extend([complex(real, imag), complex(real, -imag)])

            else:  # real value
                eigen_vals.append(self.uniform_sample(*self.EIGENVALUE_RANGE))

        return eigen_vals

    # produce simulated "measurements" of the systems
    def measure(self, f: np.ndarray, snr: float = np.inf) -> np.ndarray:

        w = 2 * np.pi * np.array(f)
        sig = signal.freqs(*self.tf, worN=w)[1]

        noise = self.uniform_sample(-1, 1, n=w.size) * np.abs(sig) / snr
        return sig + noise


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


def sparsity_check(tf: Tuple[np.ndarray, np.ndarray]) -> None:
    num, den = tf

    gain_num = num[-1]
    num = num / gain_num

    gain_den = den[-1]
    den = den / gain_den

    gain = gain_num / gain_den
    print(
        "Sparisity Check:",
        f"\tGain: {gain}",
        f"\tNum: {num}",
        f"\tDen: {den}",
        sep="\n",
    )


# generate example data
f = np.logspace(0, 7, 801, base=10)
system_n_zeros, system_n_poles = (1, 2)
model_n_zeros, model_n_poles = (3, 4)

for _ in range(3):
    # create test case
    test_system = RandomSystem(n_zeros=system_n_zeros, n_poles=system_n_poles)
    # test_system = KnownSystem(zeros=[], poles=[-1e3], gain=2.0)

    # take sample data to use to fit a model
    h = test_system.measure(f, snr=10)

    # train a model based on the example data
    model = SystemModel(max_zeros=model_n_zeros, max_poles=model_n_poles)
    try:
        model.fit(f, h, reg=1e-3)
    except ArithmeticError:
        print("\nModel failed to converge to a solution\n")
        continue

    # print comparison of models
    print(f"Test system:\n\t{test_system}")
    print(f"System model:\n\t{model}\n")
    sparsity_check(model.tf)

    # evaluate the model, plot a comparison of the fit data and the model
    # output, display any metrics
    h_model_prediction = model.predict(f)

    multiseries_bodeplot(
        Series(f, "Frequency (Hz)"),
        Series(h, "Test System", plot_type="scatter", alpha=0.8),
        Series(h_model_prediction, "Model", plot_type="scatter", alpha=0.8),
        title="Model Fitting",
        xscale="log",
    )
