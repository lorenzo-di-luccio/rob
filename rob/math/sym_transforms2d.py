import sympy
from typing import Tuple

def sym_no_rot2() -> sympy.Matrix:
    return sympy.Matrix([
        [1, 0],
        [0, 1]
    ])

def sym_hom_no_rot2() -> sympy.Matrix:
    return sympy.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

def sym_rot2(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, -s],
        [s, c]
    ])

def sym_rot2_inv(rotmat: sympy.Matrix) -> sympy.Matrix:
    return sympy.transpose(rotmat)

def sym_hom_rot2(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def sym_hom_rot2_inv(homrotmat: sympy.Matrix) -> sympy.Matrix:
    rot2_inv_mat = sympy.transpose(homrotmat[0:2, 0:2])
    return sympy.Matrix([
        [rot2_inv_mat[0, 0], rot2_inv_mat[0, 1], 0],
        [rot2_inv_mat[1, 0], rot2_inv_mat[1, 1], 0],
        [0, 0, 1]
    ])

def sym_no_transl2() -> sympy.Matrix:
    return sympy.Matrix([[0, 0]])

def sym_transl2(tx: sympy.NumberSymbol, ty: sympy.NumberSymbol) -> sympy.Matrix:
    return sympy.Matrix([[tx, ty]])

def sym_no_coltransl2() -> sympy.Matrix:
    return sympy.Matrix([0, 0])

def sym_coltransl2(tx: sympy.NumberSymbol, ty: sympy.NumberSymbol) -> sympy.Matrix:
    return sympy.Matrix([tx, ty])

def sym_transl2_inv(translvec: sympy.Matrix) -> sympy.Matrix:
    return -translvec

def sym_hom_no_transl2() -> sympy.Matrix:
    return sympy.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

def sym_hom_transl2(tx: sympy.NumberSymbol, ty: sympy.NumberSymbol) -> sympy.Matrix:
    return sympy.Matrix([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def sym_hom_transl2_inv(homtranslmat: sympy.Matrix) -> sympy.Matrix:
    transl2_inv_vec = -homtranslmat[0:2, 2:3]
    return sympy.Matrix([
        [1, 0, transl2_inv_vec[0, 0]],
        [0, 1, transl2_inv_vec[1, 0]],
        [0, 0, 1]
    ])

def sym_hom2(rotmat: sympy.Matrix, translvec: sympy.Matrix) -> sympy.Matrix:
    translvec = translvec.reshape(2, 1)
    return sympy.Matrix([
        [rotmat[0, 0], rotmat[0, 1], translvec[0, 0]],
        [rotmat[1, 0], rotmat[1, 1], translvec[1, 0]],
        [0, 0, 1]
    ])

def sym_hom2_inv(hommat: sympy.Matrix) -> sympy.Matrix:
    rot2_inv_mat = sympy.transpose(hommat[0:2, 0:2])
    transl2_inv_vec = sympy.simplify(-rot2_inv_mat * hommat[0:2, 2:3])
    return sympy.Matrix([
        [rot2_inv_mat[0, 0], rot2_inv_mat[0, 1], transl2_inv_vec[0, 0]],
        [rot2_inv_mat[1, 0], rot2_inv_mat[1, 1], transl2_inv_vec[1, 0]],
        [0, 0, 1]
    ])

def sym_dehom2(hommat: sympy.Matrix) -> Tuple[sympy.Matrix, sympy.Matrix]:
    rotmat = sympy.Matrix([
        [hommat[0, 0], hommat[0, 1]],
        [hommat[1, 0], hommat[1, 1]],
    ])
    translvec = sympy.Matrix([[hommat[0, 2], hommat[1, 2]]])
    return rotmat, translvec
