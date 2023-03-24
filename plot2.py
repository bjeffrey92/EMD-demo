from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from plot1 import plot1_data
from utils import smooth_function


def plot2_data() -> Tuple[np.ndarray]:
    x, y, y_2 = plot1_data(smooth_function, sin_amplitude=200, sin_exponent=5)
    y_3 = y_2 + np.sin(x**50) * 20
    return y, y_2, y_3


def plot(
    data: np.ndarray, imf1: np.ndarray, imf2: np.ndarray, residual: np.ndarray
) -> Tuple[np.ndarray]:
    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(data, color="orange")
    axs[0].title.set_text("Data")
    axs[1].plot(imf1)
    axs[1].title.set_text("IMF1")
    axs[2].plot(imf2)
    axs[2].title.set_text("IMF2")
    axs[3].plot(residual, color="green")
    axs[3].title.set_text("Residual")
    fig.tight_layout()
    fig.show()


def main():
    y, y_2, y_3 = plot2_data()
    data = y_3
    imf1 = y_3 - y_2
    imf2 = y_2 - y
    residual = y
    plot(data, imf1, imf2, residual)
