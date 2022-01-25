from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tkinter.filedialog import askopenfile
import re


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


def mag_phase_to_complex(magnitude: np.ndarray,
                         phase: np.ndarray,
                         mag_units='db', phase_units='deg') -> np.ndarray:

    i = complex(0, 1)

    # handle units
    if mag_units.lower() == 'db':
        mag = 10**(magnitude/20)
    else:
        mag = magnitude

    if phase_units.lower() == 'deg':
        ph = phase*(np.pi/180)
    else:
        ph = phase

    return mag*np.cos(ph) + i*mag*np.sin(ph)


def load_data(file_path: Optional[Path] = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads frequency, magnitude, and phase data from a csv file.
    If a file path is not passed as arguement the user will prompted to select
    a file throught the built-in file explorer.

    expected file format is:
    frequency_data1,magnitude_data1,phase_data1\n
    frequency_data2,magnitude_data2,phase_data2\n
    ...
    """

    # read in data
    if file_path is None:
        with askopenfile(mode='r', filetypes=[('CSV', '*.csv')],
                         defaultextension='csv') as file:
            file_contents = file.read()
    else:
        with open(file_path, mode='r') as file:
            file_contents = file.read()

    # parse file for regex matches
    signed_float = r'([+-]?\d*\.?\d*)'
    pattern = re.compile(f"{','.join([signed_float]*3)}\n")
    matches = re.findall(pattern, file_contents)

    # pack data into arrays
    f = np.zeros(len(matches))
    mag = np.zeros_like(f)
    phase = np.zeros_like(f)

    for i, match in enumerate(matches):
        data = tuple(map(float, match))
        f[i] = data[0]
        mag[i] = data[1]
        phase[i] = data[2]

    return f, mag, phase
