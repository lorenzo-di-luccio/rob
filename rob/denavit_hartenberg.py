import numpy
import sympy
from typing import *
from .position_orientation import concat_homs, hom, sym_hom, rotx, sym_rotx, rotz, sym_rotz, \
    transl, sym_transl
from .robotics_math import INFINITY

def DH_transformation(alpha: float, a: float, d: float, theta: float) -> numpy.ndarray:
    transf1 = hom(rotz(theta), transl(0., 0., d))
    transf2 = hom(rotx(alpha), transl(a, 0., 0.))
    return transf1 @ transf2

def sym_DH_transformation(alpha: sympy.NumberSymbol, a: sympy.NumberSymbol,
                          d: sympy.NumberSymbol, theta: sympy.NumberSymbol) -> sympy.Matrix:
    transf1 = sym_hom(sym_rotz(theta), sym_transl(0., 0., d))
    transf2 = sym_hom(sym_rotx(alpha), sym_transl(a, 0., 0.))
    return sympy.simplify(transf1 * transf2)

class DH_Link():
    def __init__(self, alpha: float=0., a: float=0.,
                 d: float=0., theta: float=0.,
                 min_qlim: float=None, max_qlim: float=None, type: int=0) -> None:
        self.alpha = alpha
        self.a = a
        self.d = d
        self.theta = theta
        self.min_qlim = min_qlim
        self.max_qlim = max_qlim
        self.type = type
    
    def is_joint_min_limit(self, q: float) -> bool:
        return q < self.min_qlim if self.min_qlim is not None else False
    
    def is_joint_max_limit(self, q: float) -> bool:
        return q > self.max_qlim if self.max_qlim is not None else False
    
    def is_joint_limit(self, q: float) -> bool:
        return (q < self.min_qlim if self.min_qlim is not None else False) or \
            (q > self.max_qlim if self.max_qlim is not None else False)
    
    def is_revolute(self) -> bool:
        return self.type == 0
    
    def is_prismatic(self) -> bool:
        return self.type == 1