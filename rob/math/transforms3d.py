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

def no_rot() -> numpy.ndarray:
    return numpy.eye(3, 3, dtype=numpy.float64)

def hom_no_rot() -> numpy.ndarray:
    return numpy.eye(4, 4, dtype=numpy.float64)

def rot(axis: numpy.ndarray, angle: float) -> numpy.ndarray:
    if axis.shape != (3,) or axis.shape != (3, 1):
        return numpy.eye(3, 3, dtype=numpy.float64)
    axis = axis.reshape((-1,))
    norm = numpy.linalg.norm(axis)
    axis = axis / norm if norm > EPS else axis 
    axis = axis.reshape((-1, 1))
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    axis2 = axis @ axis.T
    axis = axis.reshape((-1,))
    skewaxis = numpy.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=numpy.float64)
    return axis2 + (numpy.eye(3, 3, dtype=numpy.float64) - axis2) * c + skewaxis * s

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

def isrot(mat: numpy.ndarray) -> bool:
    shape = mat.shape
    if shape != (3, 3):
        return False
    norm = numpy.linalg.norm(mat, axis=0)
    if not numpy.all(numpy.abs(norm - numpy.ones_like(norm)) < EPS):
        return False
    if not numpy.abs(numpy.linalg.det(mat) - 1.) < EPS:
        return False
    return True

def hom_rotx(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def hom_roty(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def hom_rotz(angle: float) -> numpy.ndarray:
    c = numpy.cos(angle, dtype=numpy.float64)
    s = numpy.sin(angle, dtype=numpy.float64)
    return numpy.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def hom_rot_inv(homrotmat: numpy.ndarray) -> numpy.ndarray:
    rot_inv_mat = numpy.transpose(homrotmat[0:3, 0:3])
    return numpy.array([
        [rot_inv_mat[0, 0], rot_inv_mat[0, 1], rot_inv_mat[0, 2], 0],
        [rot_inv_mat[1, 0], rot_inv_mat[1, 1], rot_inv_mat[1, 2], 0],
        [rot_inv_mat[2, 0], rot_inv_mat[2, 1], rot_inv_mat[2, 2], 0],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def no_transl() -> numpy.ndarray:
    return numpy.zeros((3,), dtype=numpy.float64)

def transl(tx: float, ty: float, tz: float) -> numpy.ndarray:
    return numpy.array([tx, ty, tz], dtype=numpy.float64)

def no_coltransl() -> numpy.ndarray:
    return numpy.zeros((3, 1), dtype=numpy.float64)

def coltransl(tx: float, ty: float, tz: float) -> numpy.ndarray:
    return numpy.array([[tx], [ty], [tz]], dtype=numpy.float64)

def transl_inv(translvec: numpy.ndarray) -> numpy.ndarray:
    return -translvec

def hom_transl(tx: float, ty: float, tz: float) -> numpy.ndarray:
    return numpy.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def hom_transl_inv(homtranslmat: numpy.ndarray) -> numpy.ndarray:
    transl_inv_vec = -homtranslmat[0:3, 3:4]
    return numpy.array([
        [1, 0, 0, transl_inv_vec[0, 0]],
        [0, 1, 0, transl_inv_vec[1, 0]],
        [0, 0, 1, transl_inv_vec[2, 0]],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def DH_transformation(alpha: float, a: float, d: float, theta: float) -> numpy.ndarray:
    calpha = numpy.cos(alpha, dtype=numpy.float64)
    salpha = numpy.sin(alpha, dtype=numpy.float64)
    ctheta = numpy.cos(theta, dtype=numpy.float64)
    stheta = numpy.sin(theta, dtype=numpy.float64)
    transf1 = numpy.array([
        [ctheta, -stheta, 0, 0],
        [stheta, ctheta, 0, 0],
        [0, 0, 0, d],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)
    transf2 = numpy.array([
        [0, 0, 0, a],
        [0, calpha, -salpha, 0],
        [0, salpha, calpha, 0],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)
    return transf1 @ transf2

def hom(rotmat: numpy.ndarray, translvec: numpy.ndarray) -> numpy.ndarray:
    translvec = translvec.reshape((-1, 1))
    return numpy.array([
        [rotmat[0, 0], rotmat[0, 1], rotmat[0, 2], translvec[0, 0]],
        [rotmat[1, 0], rotmat[1, 1], rotmat[1, 2], translvec[1, 0]],
        [rotmat[2, 0], rotmat[2, 1], rotmat[2, 2], translvec[2, 0]],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def hom_inv(hommat: numpy.ndarray) -> numpy.ndarray:
    rot_inv_mat = numpy.transpose(hommat[0:3, 0:3])
    transl_inv_vec = -rot_inv_mat @ hommat[0:3, 3:4]
    return numpy.array([
        [rot_inv_mat[0, 0], rot_inv_mat[0, 1], rot_inv_mat[0, 2], transl_inv_vec[0, 0]],
        [rot_inv_mat[1, 0], rot_inv_mat[1, 1], rot_inv_mat[1, 2], transl_inv_vec[1, 0]],
        [rot_inv_mat[2, 0], rot_inv_mat[2, 1], rot_inv_mat[2, 2], transl_inv_vec[2, 0]],
        [0, 0, 0, 1]
    ], dtype=numpy.float64)

def ishom(mat: numpy.ndarray) -> bool:
    shape = mat.shape
    if shape != (4, 4):
        return False
    submat = mat[0:3, 0:3]
    norm = numpy.linalg.norm(submat, axis=0)
    if not numpy.all(numpy.abs(norm - numpy.ones_like(norm)) < EPS):
        return False
    if not numpy.abs(numpy.linalg.det(submat) - 1.) < EPS:
        return False
    return True

def dehom(hommat: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    rotmat = numpy.array([
        [hommat[0, 0], hommat[0, 1], hommat[0, 2]],
        [hommat[1, 0], hommat[1, 1], hommat[1, 2]],
        [hommat[2, 0], hommat[2, 1], hommat[2, 2]]
    ], dtype=numpy.float64)
    translvec = numpy.array([hommat[0, 3], hommat[1, 3], hommat[2, 3]], dtype=numpy.float64)
    return rotmat, translvec
