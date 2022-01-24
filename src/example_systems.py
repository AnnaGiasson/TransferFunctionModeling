from dataclasses import dataclass
from numbers import Number
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np
from scipy import signal


class System(Protocol):
    def __str__(self) -> str:
        """
        returns a string with the name of the class in addition to its transfer
        function, in the form of its zeros, poles, and gain
        """

    def measure(self, f: np.ndarray) -> np.ndarray:
        """Samples the frequency response of the system at frequencies "f" """


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
