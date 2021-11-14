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

def rotx(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=numpy.float64)

def roty(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=numpy.float64)

def rotz(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=numpy.float64)

def rot_inv(rotmat: numpy.ndarray) -> numpy.ndarray:
    return numpy.array(numpy.transpose(rotmat), copy=True)