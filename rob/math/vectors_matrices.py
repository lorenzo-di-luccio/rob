import numpy
import numpy.linalg
from typing import Iterable

from .base import EPS

def vec(v: Iterable[float]) -> numpy.ndarray:
    return numpy.array(v, dtype=numpy.float64)

def colvec(v: Iterable[float]) -> numpy.ndarray:
    return numpy.array(v, dtype=numpy.float64).reshape((-1, 1))

def unitvec(v: Iterable[float]) -> numpy.ndarray:
    vec = numpy.array(v, dtype=numpy.float64)
    norm = numpy.linalg.norm(vec)
    return vec / norm if norm > EPS else vec

def colunitvec(v: Iterable[float]) -> numpy.ndarray:
    vec = numpy.array(v, dtype=numpy.float64)
    norm = numpy.linalg.norm(vec)
    return (vec / norm if norm > EPS else vec).reshape((-1, 1))

def isunitvec(vec: numpy.ndarray) -> bool:
    return numpy.abs(numpy.linalg.norm(vec) - 1.) < EPS

def iszerovec(vec: numpy.ndarray) -> bool:
    return numpy.linalg.norm(vec) < EPS
