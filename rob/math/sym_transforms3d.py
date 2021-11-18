import sympy
from typing import Tuple

def sym_skew(vec: sympy.Matrix) -> sympy.Matrix:
    vec = vec.reshape(1, 3)
    return sympy.Matrix([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

def sym_vex(skewmat: sympy.Matrix) -> sympy.Matrix:
    return sympy.Matrix([skewmat[2, 1], skewmat[0, 2], skewmat[1, 0]])

def sym_no_rot() -> sympy.Matrix:
    return sympy.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

def sym_hom_no_rot() -> sympy.Matrix:
    return sympy.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def sym_rot(axis: sympy.Matrix, angle: sympy.NumberSymbol) -> sympy.Matrix:
    if axis.shape != (1, 3) and axis.shape != (3, 1):
        return sympy.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    axis = axis.reshape(3, 1)
    axis = sympy.simplify(axis.normalized())
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    axis2 = sympy.simplify(axis * axis.T)
    axis = axis.reshape(1, 3)
    skewaxis = sympy.Matrix([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return sympy.simplify(axis2 + (sympy.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]) - axis2) * c + skewaxis * s)

def sym_rotx(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def sym_roty(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def sym_rotz(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def sym_rot_eulzxz(anglez: sympy.NumberSymbol, anglex: sympy.NumberSymbol, anglez1: sympy.NumberSymbol) \
    -> sympy.Matrix:
    cz = sympy.cos(anglez)
    sz = sympy.sin(anglez)
    cx = sympy.cos(anglex)
    sx = sympy.sin(anglex)
    cz1 = sympy.cos(anglez1)
    sz1 = sympy.sin(anglez1)
    rotzmat = sympy.Matrix([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])
    rotxmat = sympy.Matrix([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    rotz1mat = sympy.Matrix([
        [cz1, -sz1, 0],
        [sz1, cz1, 0],
        [0, 0, 1]
    ])
    return sympy.simplify(rotzmat * rotxmat * rotz1mat)

def sym_rot_rpyxyz(anglex: sympy.NumberSymbol, angley: sympy.NumberSymbol, anglez: sympy.NumberSymbol) \
    -> sympy.Matrix:
    cx = sympy.cos(anglex)
    sx = sympy.sin(anglex)
    cy = sympy.cos(angley)
    sy = sympy.sin(angley)
    cz = sympy.cos(anglez)
    sz = sympy.sin(anglez)
    rotxmat = sympy.Matrix([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    rotymat = sympy.Matrix([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    rotzmat = sympy.Matrix([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])
    return sympy.simplify(rotzmat * rotymat * rotxmat)

def sym_rot_inv(rotmat: sympy.Matrix) -> sympy.Matrix:
    return sympy.transpose(rotmat)

def sym_hom_rotx(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def sym_hom_roty(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def sym_hom_rotz(angle: sympy.NumberSymbol) -> sympy.Matrix:
    c = sympy.cos(angle)
    s = sympy.sin(angle)
    return sympy.Matrix([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def sym_hom_rot_inv(homrotmat: sympy.Matrix) -> sympy.Matrix:
    rot_inv_mat = sympy.transpose(homrotmat[0:3, 0:3])
    return sympy.Matrix([
        [rot_inv_mat[0, 0], rot_inv_mat[0, 1], rot_inv_mat[0, 2], 0],
        [rot_inv_mat[1, 0], rot_inv_mat[1, 1], rot_inv_mat[1, 2], 0],
        [rot_inv_mat[2, 0], rot_inv_mat[2, 1], rot_inv_mat[2, 2], 0],
        [0, 0, 0, 1]
    ])

def sym_no_transl() -> sympy.Matrix:
    return sympy.Matrix([[0, 0, 0]])

def sym_transl(tx: sympy.NumberSymbol, ty: sympy.NumberSymbol, tz: sympy.NumberSymbol) -> sympy.Matrix:
    return sympy.Matrix([[tx, ty, tz]])

def sym_no_coltransl() -> sympy.Matrix:
    return sympy.Matrix([0, 0, 0])

def sym_coltransl(tx: sympy.NumberSymbol, ty: sympy.NumberSymbol, tz: sympy.NumberSymbol) -> sympy.Matrix:
    return sympy.Matrix([tx, ty, tz])

def sym_transl_inv(translvec: sympy.Matrix) -> sympy.Matrix:
    return -translvec

def sym_hom_transl(tx: sympy.NumberSymbol, ty: sympy.NumberSymbol, tz: sympy.NumberSymbol) -> sympy.Matrix:
    return sympy.Matrix([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def sym_hom_transl_inv(homtranslmat: sympy.Matrix) -> sympy.Matrix:
    transl_inv_vec = -homtranslmat[0:3, 3:4]
    return sympy.Matrix([
        [1, 0, 0, transl_inv_vec[0, 0]],
        [0, 1, 0, transl_inv_vec[1, 0]],
        [0, 0, 1, transl_inv_vec[2, 0]],
        [0, 0, 0, 1]
    ])

def sym_DH_transformation(alpha: sympy.NumberSymbol, a: sympy.NumberSymbol,
                          d: sympy.NumberSymbol, theta: sympy.NumberSymbol) -> sympy.Matrix:
    calpha = sympy.cos(alpha)
    salpha = sympy.sin(alpha)
    ctheta = sympy.cos(theta)
    stheta = sympy.sin(theta)
    transf1 = sympy.Matrix([
        [ctheta, -stheta, 0, 0],
        [stheta, ctheta, 0, 0],
        [0, 0, 0, d],
        [0, 0, 0, 1]
    ])
    transf2 = sympy.Matrix([
        [0, 0, 0, a],
        [0, calpha, -salpha, 0],
        [0, salpha, calpha, 0],
        [0, 0, 0, 1]
    ])
    return sympy.simplify(transf1 * transf2)

def sym_hom(rotmat: sympy.Matrix, translvec: sympy.Matrix) -> sympy.Matrix:
    translvec = translvec.reshape(3, 1)
    return sympy.Matrix([
        [rotmat[0, 0], rotmat[0, 1], rotmat[0, 2], translvec[0, 0]],
        [rotmat[1, 0], rotmat[1, 1], rotmat[1, 2], translvec[1, 0]],
        [rotmat[2, 0], rotmat[2, 1], rotmat[2, 2], translvec[2, 0]],
        [0, 0, 0, 1]
    ])

def sym_hom_inv(hommat: sympy.Matrix) -> sympy.Matrix:
    rot_inv_mat = sympy.transpose(hommat[0:3, 0:3])
    transl_inv_vec = sympy.simplify(-rot_inv_mat * hommat[0:3, 3:4])
    return sympy.Matrix([
        [rot_inv_mat[0, 0], rot_inv_mat[0, 1], rot_inv_mat[0, 2], transl_inv_vec[0, 0]],
        [rot_inv_mat[1, 0], rot_inv_mat[1, 1], rot_inv_mat[1, 2], transl_inv_vec[1, 0]],
        [rot_inv_mat[2, 0], rot_inv_mat[2, 1], rot_inv_mat[2, 2], transl_inv_vec[2, 0]],
        [0, 0, 0, 1]
    ])

def dehom(hommat: sympy.Matrix) -> Tuple[sympy.Matrix, sympy.Matrix]:
    rotmat = sympy.Matrix([
        [hommat[0, 0], hommat[0, 1], hommat[0, 2]],
        [hommat[1, 0], hommat[1, 1], hommat[1, 2]],
        [hommat[2, 0], hommat[2, 1], hommat[2, 2]]
    ])
    translvec = sympy.Matrix([[hommat[0, 3], hommat[1, 3], hommat[2, 3]]])
    return rotmat, translvec
