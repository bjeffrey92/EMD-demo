from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import argrelextrema


N = 15


def smooth_function(x: Union[int, float, np.array]) -> Union[int, float, np.array]:
    return (x**3 - 10 * x**2 + 100 + 0.001 * x**4) / 2


def plot1_data(
    smooth_function: Callable,
    sin_exponent: int = 5,
    sin_amplitude: int = 100,
    plot: bool = False,
) -> Tuple[np.array]:
    x = np.arange(0, N, 1)
    y = smooth_function(x)

    noisey_y = np.sin(x**sin_exponent) * sin_amplitude + y

    s = interpolate.InterpolatedUnivariateSpline(x, y)
    xfit = np.arange(0, N - 1, np.pi / 50)
    yfit = s(xfit)

    s_noisey = interpolate.InterpolatedUnivariateSpline(x, noisey_y)
    yfit_with_noise = s_noisey(xfit)

    if plot:
        plt.clf()
        plt.plot(xfit, yfit)
        plt.plot(xfit, yfit_with_noise)
        plt.show()
    else:
        return xfit, yfit, yfit_with_noise


def compute_envelopes(y_with_noise: np.array) -> Tuple[np.array]:
    idx_local_maxima = argrelextrema(y_with_noise, np.greater)
    idx_local_minima = argrelextrema(y_with_noise, np.less)

    def envelope(idx_extrema: np.array) -> np.array:
        extrema = y_with_noise[idx_extrema]

        x = idx_extrema[0]
        y = extrema
        s = interpolate.InterpolatedUnivariateSpline(x, y)
        yfit = s(np.arange(0, len(y_with_noise) + 2))
        return extrema, yfit

    return *envelope(idx_local_maxima), *envelope(idx_local_minima)


def main():
    x, y, y_with_noise = plot1_data(smooth_function, sin_amplitude=200, sin_exponent=5)
    local_maxima, upper_envelope, local_minima, lower_envelope = compute_envelopes(
        y_with_noise
    )
