import numpy
import numpy.linalg
from typing import Tuple

from .base import EPS

def no_rot2() -> numpy.ndarray:
    return numpy.eye(2, 2, dtype=numpy.float64)

def hom_no_rot2() -> numpy.ndarray:
    return numpy.eye(3, 3, dtype=numpy.float64)

def rot2(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, -s],
        [s, c]
    ], dtype=numpy.float64)

def rot2_inv(rotmat: numpy.ndarray) -> numpy.ndarray:
    return numpy.array(numpy.transpose(rotmat), copy=True)

def hom_rot2(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=numpy.float64)

def hom_rot2_inv(homrotmat: numpy.ndarray) -> numpy.ndarray:
    rot2_inv_mat = numpy.transpose(homrotmat[0:2, 0:2])
    return numpy.array([
        [rot2_inv_mat[0, 0], rot2_inv_mat[0, 1], 0],
        [rot2_inv_mat[1, 0], rot2_inv_mat[1, 1], 0],
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

def no_transl2() -> numpy.ndarray:
    return numpy.zeros((2,), dtype=numpy.float64)

def transl2(tx: float, ty: float) -> numpy.ndarray:
    return numpy.array([tx, ty], dtype=numpy.float64)

def no_coltransl2() -> numpy.ndarray:
    return numpy.zeros((2, 1), dtype=numpy.float64)

def coltransl2(tx: float, ty: float) -> numpy.ndarray:
    return numpy.array([[tx], [ty]], dtype=numpy.float64)

def transl2_inv(translvec: numpy.ndarray) -> numpy.ndarray:
    return -translvec

def hom_no_transl2() -> numpy.ndarray:
    return numpy.eye(3, 3, dtype=numpy.float64) 

def hom_transl2(tx: float, ty: float) -> numpy.ndarray:
    return numpy.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=numpy.float64)

def hom_transl2_inv(homtranslmat: numpy.ndarray) -> numpy.ndarray:
    transl2_inv_vec = -homtranslmat[0:2, 2:3]
    return numpy.array([
        [1, 0, transl2_inv_vec[0, 0]],
        [0, 1, transl2_inv_vec[1, 0]],
        [0, 0, 1]
    ], dtype=numpy.float64)

def hom2(rotmat: numpy.ndarray, translvec: numpy.ndarray) -> numpy.ndarray:
    translvec = translvec.reshape((-1, 1))
    return numpy.array([
        [rotmat[0, 0], rotmat[0, 1], translvec[0, 0]],
        [rotmat[1, 0], rotmat[1, 1], translvec[1, 0]],
        [0, 0, 1]
    ], dtype=numpy.float64)

def hom2_inv(hommat: numpy.ndarray) -> numpy.ndarray:
    rot2_inv_mat = numpy.transpose(hommat[0:2, 0:2])
    transl2_inv_vec = -rot2_inv_mat @ hommat[0:2, 2:3]
    return numpy.array([
        [rot2_inv_mat[0, 0], rot2_inv_mat[0, 1], transl2_inv_vec[0, 0]],
        [rot2_inv_mat[1, 0], rot2_inv_mat[1, 1], transl2_inv_vec[1, 0]],
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
