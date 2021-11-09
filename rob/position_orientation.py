import math
import numpy
import sympy
from typing import *

def skew(vec: numpy.ndarray) -> numpy.ndarray:
    return numpy.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ], dtype=float)

def sym_skew(vec: sympy.Matrix) -> sympy.Matrix:
    return sympy.Matrix([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

def rot(axis: numpy.ndarray, angle: float) -> numpy.ndarray:
    new_axis = numpy.reshape(axis, (3, 1))
    axis_mat = new_axis @ new_axis.T
    axis_skew_mat = numpy.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=float)
    c = math.cos(angle)
    s = math.sin(angle)
    return axis_mat + (numpy.eye(3, 3) - axis_mat) * c + axis_skew_mat * s

def sym_rot(axis: sympy.Matrix, angle: sympy.NumberSymbol) -> sympy.Matrix:
    axis_mat = axis * axis.T
    axis_skew_mat = sympy.Matrix([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.simplify(axis_mat + (sympy.Identity(3) - axis_mat) * c + axis_skew_mat * s)

def rotx(angle: float) -> numpy.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return numpy.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=float)

def sym_rotx(angle: sympy.NumberSymbol) -> numpy.ndarray:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def roty(angle: float) -> numpy.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return numpy.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=float)

def sym_roty(angle: sympy.NumberSymbol) -> numpy.ndarray:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotz(angle: float) -> numpy.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return numpy.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)

def sym_rotz(angle: sympy.NumberSymbol) -> numpy.ndarray:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def rot_rpy(anglex: float, angley: float, anglez: float) -> numpy.ndarray:
    cx = math.cos(anglex)
    sx = math.sin(anglex)
    cy = math.cos(angley)
    sy = math.sin(angley)
    cz = math.cos(anglez)
    sz = math.sin(anglez)
    rotzm = numpy.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=float)
    rotym = numpy.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=float)
    rotxm = numpy.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=float)
    return rotzm @ rotym @ rotxm

def sym_rot_rpy(anglex: sympy.NumberSymbol, angley: sympy.NumberSymbol, anglez: sympy.NumberSymbol) \
    -> sympy.Matrix:
    cx = sympy.cos(anglex)
    sx = sympy.sin(anglex)
    cy = sympy.cos(angley)
    sy = sympy.sin(angley)
    cz = sympy.cos(anglez)
    sz = sympy.sin(anglez)
    rotzm = sympy.Matrix([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=float)
    rotym = sympy.Matrix([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=float)
    rotxm = sympy.Matrix([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=float)
    return sympy.simplify(rotzm * rotym * rotxm)

def concat_rot(rot1: numpy.ndarray, rot2: numpy.ndarray) -> numpy.ndarray:
    return rot1 @ rot2

def sym_concat_rot(rot1: sympy.Matrix, rot2: sympy.Matrix) -> sympy.Matrix:
    return sympy.simplify(rot1 * rot2)

def concat_rots(rots: Iterable[numpy.ndarray]) -> numpy.ndarray:
    total_rot = numpy.eye(3, 3, dtype=float)
    for rot in rots:
        total_rot @= rot
    return total_rot

def sym_concat_rots(rots: Iterable[sympy.Matrix]) -> sympy.Matrix:
    total_rot = sympy.Identity(3)
    for rot in rots:
        total_rot *= rot
    return sympy.simplify(total_rot)

def inverse_rot(rot: numpy.ndarray) -> numpy.ndarray:
    return numpy.transpose(rot)

def sym_inverse_rot(rot: sympy.Matrix) -> sympy.Matrix:
    return sympy.transpose(rot)

def transl(x: float, y: float, z: float) -> numpy.ndarray:
    return numpy.array([x, y, z], dtype=float)

def sym_transl(x: sympy.NumberSymbol, y: sympy.NumberSymbol, z: sympy.NumberSymbol) -> sympy.Matrix:
    return sympy.Matrix([x, y, z])

def concat_transl(transl1: numpy.ndarray, transl2: numpy.ndarray) -> numpy.ndarray:
    return transl1 + transl2

def sym_concat_transl(transl1: sympy.Matrix, transl2: sympy.Matrix) -> sympy.Matrix:
    return sympy.simplify(transl1 + transl2)

def concat_transls(transls: Iterable[numpy.ndarray]) -> numpy.ndarray:
    total_transl = numpy.zeros((3,), dtype=float)
    for transl in transls:
        total_transl += transl
    return total_transl

def sym_concat_transls(transls: Iterable[sympy.Matrix]) -> sympy.Matrix:
    total_transl = sympy.ZeroMatrix(3, 1)
    for transl in transls:
        total_transl += transl
    return sympy.simplify(total_transl)

def inverse_transl(transl: numpy.ndarray) -> numpy.ndarray:
    return -transl

def sym_inverse_transl(transl: sympy.Matrix) -> sympy.Matrix:
    return -transl

def hom(rot: numpy.ndarray, transl: numpy.ndarray) -> numpy.ndarray:
    transl = numpy.reshape(transl, (3, 1))
    homm = numpy.eye(4, 4, dtype=float)
    homm[0:3, 0:3] = rot
    homm[0:3, 3:4] = transl
    return homm

def sym_hom(rot: sympy.Matrix, transl: sympy.Matrix) -> sympy.Matrix:
    homm = sympy.Identity(4)
    homm[0:3, 0:3] = rot
    homm[0:3, 3] = transl
    return homm

def concat_hom(hom1: numpy.ndarray, hom2: numpy.ndarray) -> numpy.ndarray:
    return hom1 @ hom2

def sym_concat_hom(hom1: sympy.Matrix, hom2: sympy.Matrix) -> sympy.Matrix:
    return sympy.simplify(hom1 * hom2)

def concat_homs(homs: Iterable[numpy.ndarray]) -> numpy.ndarray:
    total_hom = numpy.eye(4, dtype=float)
    for hom in homs:
        total_hom @= hom
    return total_hom

def sym_concat_homs(homs: Iterable[sympy.Matrix]) -> sympy.Matrix:
    total_hom = sympy.Identity(4)
    for hom in homs:
        total_hom *= hom
    return sympy.simplify(total_hom)

def inverse_hom(hom: numpy.ndarray) -> numpy.ndarray:
    rot = numpy.array(hom[0:3, 0:3], dtype=float)
    inv_rot = numpy.transpose(rot)
    transl = numpy.array(hom[0:3, 3], dtype=float)
    inv_transl = -inv_rot @ transl
    inv_homm = numpy.eye(4, 4, dtype=float)
    inv_homm[0:3, 0:3] = inv_rot
    inv_homm[0:3, 3:4] = inv_transl
    return inv_homm

def sym_inverse_hom(hom: sympy.Matrix) -> sympy.Matrix:
    rot = sympy.Matrix(hom[0:3, 0:3])
    inv_rot = sympy.transpose(rot)
    transl = sympy.Matrix(hom[0:3, 3])
    inv_transl = sympy.simplify(-inv_rot * transl)
    inv_homm = sympy.Identity(4)
    inv_homm[0:3, 0:3] = inv_rot
    inv_homm[0:3, 3] = inv_transl
    return inv_homm

def dehom(hom: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    rot = numpy.array(hom[0:3, 0:3], dtype=float)
    retr_transl = numpy.array(hom[0:3, 3], dtype=float)
    transl = numpy.reshape(retr_transl, (3,))
    return rot, transl

def sym_dehom(hom: sympy.Matrix) -> Tuple[sympy.Matrix, sympy.Matrix]:
    rot = sympy.Matrix(hom[0:3, 0:3])
    transl = sympy.Matrix(hom[0:3, 3])
    return rot, transl
