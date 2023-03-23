from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def smooth_function(x: Union[int, float, np.array]) -> Union[int, float, np.array]:
    return x**3 - 7 * x**2 + 100


def plot1_data(
    smooth_function: Callable,
    n: int = 9,
    sin_exponent: int = 5,
    sin_amplitude: int = 100,
    plot: bool = False,
):
    x = np.arange(0, n, 1)
    y = smooth_function(x)

    noisey_y = np.sin(x**sin_exponent) * sin_amplitude + y

    s = interpolate.InterpolatedUnivariateSpline(x, y)
    xfit = np.arange(0, n - 1, np.pi / 50)
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


def main():
    xfit, yfit, yfit_with_noise = plot1_data(
        smooth_function, n=12, sin_amplitude=200, sin_exponent=8
    )
