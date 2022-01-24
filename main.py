from typing import Tuple

import numpy as np

from src.example_systems import RandomSystem
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


# generate example data
f = np.logspace(0, 7, 401, base=10)
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
        FrequencySeries(
            magnitude=magnitude_db(h),
            phase=phase_deg(h),
            label="Test System",
            plot_type="scatter",
            alpha=0.6,
            color="C0",
        ),
        FrequencySeries(
            magnitude=magnitude_db(h_model_prediction),
            phase=phase_deg(h_model_prediction),
            label="Model",
            plot_type="plot",
            alpha=0.9,
            color="C1",
        ),
        title="Model Fitting",
        xscale="log",
    )
