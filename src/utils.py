import numpy as np


def magnitude(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def magnitude_db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log(magnitude(x))


def phase(x: np.ndarray) -> np.ndarray:
    return np.angle(x)


def phase_deg(x: np.ndarray) -> np.ndarray:
    return phase(x) * (180 / np.pi)
