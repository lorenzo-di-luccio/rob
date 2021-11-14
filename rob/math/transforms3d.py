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

def rot_eulzxz(anglez: float, anglex: float, anglez1: float) -> numpy.ndarray:
    cz = numpy.cos(anglez, dtype=numpy.float64)
    sz = numpy.sin(anglez, dtype=numpy.float64)
    cx = numpy.cos(anglex, dtype=numpy.float64)
    sx = numpy.sin(anglex, dtype=numpy.float64)
    cz1 = numpy.cos(anglez1, dtype=numpy.float64)
    sz1 = numpy.sin(anglez1, dtype=numpy.float64)
    rotzmat = numpy.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=numpy.float64)
    rotxmat = numpy.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=numpy.float64)
    rotz1mat = numpy.array([
        [cz1, -sz1, 0],
        [sz1, cz1, 0],
        [0, 0, 1]
    ], dtype=numpy.float64)
    return rotzmat @ rotxmat @ rotz1mat

def rot_rpyxyz(anglex: float, angley: float, anglez: float) -> numpy.ndarray:
    cx = numpy.cos(anglex, dtype=numpy.float64)
    sx = numpy.sin(anglex, dtype=numpy.float64)
    cy = numpy.cos(angley, dtype=numpy.float64)
    sy = numpy.sin(angley, dtype=numpy.float64)
    cz = numpy.cos(anglez, dtype=numpy.float64)
    sz = numpy.sin(anglez, dtype=numpy.float64)
    rotxmat = numpy.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=numpy.float64)
    rotymat = numpy.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=numpy.float64)
    rotzmat = numpy.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=numpy.float64)
    return rotzmat @ rotymat @ rotxmat

def rot_inv(rotmat: numpy.ndarray) -> numpy.ndarray:
    return numpy.array(numpy.transpose(rotmat), copy=True)
