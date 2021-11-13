import numpy
import numpy.linalg
from typing import Tuple

from .base import EPS

def skew(vec: numpy.ndarray) -> numpy.ndarray:
    vec = vec.reshape((-1,))
    return numpy.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ], dtype=numpy.float64)

def vex(skewmat: numpy.ndarray) -> numpy.ndarray:
    return numpy.array([skewmat[2, 1], skewmat[0, 2], skewmat[1, 0]], dtype=numpy.float64)
