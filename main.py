from typing import Tuple

import numpy as np

from src.example_systems import RandomSystem, System
from src.model_fitting import SystemModel
from src.plotting import FrequencySeries, Series, multiseries_bodeplot
from src.utils import magnitude_db, phase_deg


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


def evaluate_model(f: np.ndarray, sys_real: System, sys_model: System) -> None:

    # print comparison of models
    print(f"Test system:\n\t{test_system}")
    print(f"System model:\n\t{model}\n")
    sparsity_check(sys_model.tf)

    # evaluate the model, plot a comparison of the fit data and the model
    # output, display any metrics
    h_test = sys_real.measure(f)
    h_pred = sys_model.measure(f)

    multiseries_bodeplot(
        Series(f, "Frequency (Hz)"),
        FrequencySeries(
            magnitude=magnitude_db(h_test),
            phase=phase_deg(h_test),
            label="Test System",
            plot_type="scatter",
            alpha=0.6,
            color="C0",
        ),
        FrequencySeries(
            magnitude=magnitude_db(h_pred),
            phase=phase_deg(h_pred),
            label="Model",
            plot_type="plot",
            alpha=0.9,
            color="C1",
        ),
        title="Model Fitting",
        xscale="log",
    )


# script config
n_tests: int = 3

system_n_zeros, system_n_poles = (1, 2)
model_n_zeros, model_n_poles = (3, 4)

f_range_start: Tuple[int] = (0, 1, 2, 3, 4)  # 10^x
f_range_end: Tuple[int] = (6, 7, 8)  # 10^x
n_points: Tuple[int] = (201, 401, 801)  # base-10 log spaced

# test loop
for _ in range(n_tests):

    # create test case / generate example data
    f_train = np.logspace(
        np.random.choice(f_range_start),
        np.random.choice(f_range_end),
        np.random.choice(n_points),
        base=10,
    )
    test_system = RandomSystem(n_zeros=system_n_zeros, n_poles=system_n_poles)
    # test_system = KnownSystem(zeros=[], poles=[-1e3], gain=2.0)

    h_train = test_system.measure(f_train, snr=10)  # training data

    # train a model based on the example data
    model = SystemModel(max_zeros=model_n_zeros, max_poles=model_n_poles)
    try:
        model.fit(f_train, h_train, reg=1e-3)
    except ArithmeticError:
        print("\nModel failed to converge to a solution\n")
        continue

    f_test = np.logspace(
        np.random.choice(f_range_start),
        np.random.choice(f_range_end),
        401,
        base=10,
    )

    evaluate_model(f_test, test_system, model)
