import numpy
import numpy.linalg
from typing import Iterable, Tuple

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

def norm(vec: numpy.ndarray) -> float:
    return numpy.linalg.norm(vec)

def sqnorm(vec: numpy.ndarray) -> float:
    return numpy.dot(vec, vec)

def isunitvec(vec: numpy.ndarray) -> bool:
    return numpy.abs(numpy.linalg.norm(vec) - 1.) < EPS

def iszerovec(vec: numpy.ndarray) -> bool:
    return numpy.linalg.norm(vec) < EPS

def rot2(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, -s],
        [s, c]
    ], dtype=numpy.float64)

def hom_rot2(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=numpy.float64)

def isrot2(mat: numpy.ndarray) -> bool:
    shape = mat.shape
    if shape != (2, 2):
        return False
    norm = numpy.linalg.norm(mat, axis=0)
    if not numpy.all(numpy.abs(norm - numpy.ones_like(norm)) < EPS):
        return False
    if not numpy.abs(numpy.linalg.det(mat) - 1.) < EPS:
        return False
    return True

def transl2(tx: float, ty: float) -> numpy.ndarray:
    return numpy.array([tx, ty], dtype=numpy.float64)

def coltransl2(tx: float, ty: float) -> numpy.ndarray:
    return numpy.array([[tx], [ty]], dtype=numpy.float64)

def hom_transl2(tx: float, ty: float) -> numpy.ndarray:
    return numpy.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=numpy.float64)

def hom2(rotmat: numpy.ndarray, translvec: numpy.ndarray) -> numpy.ndarray:
    translvec = translvec.reshape((-1, 1))
    return numpy.array([
        [rotmat[0, 0], rotmat[0, 1], translvec[0, 0]],
        [rotmat[1, 0], rotmat[1, 1], translvec[1, 0]],
        [0, 0, 1]
    ], dtype=numpy.float64)

def ishom2(mat: numpy.ndarray) -> bool:
    shape = mat.shape
    if shape != (3, 3):
        return False
    submat = mat[0:2, 0:2]
    norm = numpy.linalg.norm(submat, axis=0)
    if not numpy.all(numpy.abs(norm - numpy.ones_like(norm)) < EPS):
        return False
    if not numpy.abs(numpy.linalg.det(submat) - 1.) < EPS:
        return False
    return True

def dehom2(hommat: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    rotmat = numpy.array([
        [hommat[0, 0], hommat[0, 1]],
        [hommat[1, 0], hommat[1, 1]],
    ], dtype=numpy.float64)
    translvec = numpy.array([hommat[0, 2], hommat[1, 2]], dtype=numpy.float64)
    return rotmat, translvec