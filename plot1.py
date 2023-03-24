from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import argrelextrema


N = 15


def smooth_function(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
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


def get_extrema(y_with_noise: np.ndarray) -> Tuple[np.ndarray]:
    idx_local_maxima = argrelextrema(y_with_noise, np.greater)
    idx_local_minima = argrelextrema(y_with_noise, np.less)
    return idx_local_maxima, idx_local_minima


def compute_envelopes(
    y_with_noise: np.ndarray, idx_local_maxima: np.ndarray, idx_local_minima: np.ndarray
) -> Tuple[np.array]:
    def envelope(idx_extrema: np.ndarray) -> np.ndarray:
        extrema = y_with_noise[idx_extrema]

        x = idx_extrema[0]
        y = extrema
        s = interpolate.InterpolatedUnivariateSpline(x, y)
        yfit = s(np.arange(0, len(y_with_noise) + 2))
        return extrema, yfit

    return *envelope(idx_local_maxima), *envelope(idx_local_minima)


def plot(
    x: np.ndarray,
    component: np.ndarray,
    y_with_noise: np.ndarray,
    upper_envelope: np.ndarray,
    lower_envelope: np.ndarray,
    idx_local_maxima: np.ndarray,
    idx_local_minima: np.ndarray,
    local_maxima: np.ndarray,
    local_minima: np.ndarray,
):
    plt.clf()
    plt.plot(x[20:], component[21:-1], color="black", linestyle="--", label="Component")
    plt.plot(x[20:], y_with_noise[20:], color="blue", label="Noisey Data")
    plt.plot(
        x[20:], upper_envelope[21:-1], color="green", label="Upper & Lower Envelopes"
    )
    plt.plot(x[20:], lower_envelope[21:-1], color="green")
    plt.scatter(x[idx_local_maxima], local_maxima, label="Local Maxima", color="red")
    plt.scatter(x[idx_local_minima], local_minima, label="Local Minima", color="orange")
    plt.legend()
    plt.show()


def main():
    x, y, y_with_noise = plot1_data(smooth_function, sin_amplitude=200, sin_exponent=5)
    idx_local_maxima, idx_local_minima = get_extrema(y_with_noise)
    local_maxima, upper_envelope, local_minima, lower_envelope = compute_envelopes(
        y_with_noise, idx_local_maxima, idx_local_minima
    )
    component = np.mean((lower_envelope, upper_envelope), axis=0)
    plot(
        x,
        component,
        y_with_noise,
        upper_envelope,
        lower_envelope,
        idx_local_maxima,
        idx_local_minima,
        local_maxima,
        local_minima,
    )
