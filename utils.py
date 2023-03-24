from typing import Union

import numpy as np

N = 15


def smooth_function(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    return (x**3 - 10 * x**2 + 100 + 0.001 * x**4) / 2
