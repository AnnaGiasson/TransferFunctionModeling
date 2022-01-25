import numpy as np

from src.model_fitting import SystemModel
from src.plotting import FrequencySeries, Series, multiseries_bodeplot
from src.utils import magnitude_db, phase_deg, mag_phase_to_complex, load_data


def plot_comparison(f: np.ndarray, h_real: np.ndarray,
                    h_model: np.ndarray) -> None:

    # plot a comparison of the fit data and the model output, display any
    # metrics
    multiseries_bodeplot(
        Series(f, "Frequency (Hz)"),
        FrequencySeries(
            magnitude=magnitude_db(h_real),
            phase=phase_deg(h_real),
            label="Test System",
            plot_type="scatter",
            alpha=0.6,
            color="C0",
        ),
        FrequencySeries(
            magnitude=magnitude_db(h_model),
            phase=phase_deg(h_model),
            label="Model",
            plot_type="plot",
            alpha=0.9,
            color="C1",
        ),
        title="Model Fitting",
        xscale="log",
    )


# script config
n_zeros, n_poles = (1, 3)

# load data to fit
f_measurement, magnitude, phase = load_data()
h_measurement = mag_phase_to_complex(magnitude, phase,
                                     mag_units='db', phase_units='deg')

# train a model based on the example data
model = SystemModel(max_zeros=n_zeros, max_poles=n_poles)
try:
    model.fit(f_measurement, h_measurement, reg=1e-3)
except ArithmeticError:
    print("\nModel failed to converge to a solution\n")
    quit()

print(f'Model: {model}')

h_pred = model.predict(f_measurement)
plot_comparison(f_measurement, h_measurement, h_pred)
