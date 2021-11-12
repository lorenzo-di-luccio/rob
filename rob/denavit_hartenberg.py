import numpy
import sympy
from typing import *
from .position_orientation import concat_homs, hom, sym_hom, rotx, sym_rotx, rotz, sym_rotz, transl, sym_transl

def DH_transformation(alpha: float, a: float, d: float, theta: float) -> numpy.ndarray:
    transf1 = hom(rotz(theta), transl(0., 0., d))
    transf2 = hom(rotx(alpha), transl(a, 0., 0.))
    return transf1 @ transf2

def sym_DH_transformation(alpha: sympy.NumberSymbol, a: sympy.NumberSymbol,
                          d: sympy.NumberSymbol, theta: sympy.NumberSymbol) -> sympy.Matrix:
    transf1 = sym_hom(sym_rotz(theta), sym_transl(0., 0., d))
    transf2 = sym_hom(sym_rotx(alpha), sym_transl(a, 0., 0.))
    return sympy.simplify(transf1 * transf2)

class DH_Joint():
    def __init__(self, type: str,
                 alpha: float, a: float,
                 d: float, theta: float,
                 sym_alpha: sympy.NumberSymbol, sym_a: sympy.NumberSymbol,
                 sym_d: sympy.NumberSymbol, sym_theta: sympy.NumberSymbol) -> None:
        self.type = type
        self.alpha = alpha
        self.a = a
        self.d = d
        self.theta = theta
        self.sym_alpha = sym_alpha
        self.sym_a = sym_a
        self.sym_d = sym_d
        self.sym_theta = sym_theta
    
    def set_params(self,
                   alpha: float, a: float,
                   d: float, theta: float) -> None:
        self.alpha = alpha
        self.a = a
        self.d = d
        self.theta = theta
    
    def set_sym_params(self,
                       sym_alpha: sympy.NumberSymbol, sym_a: sympy.NumberSymbol,
                       sym_d: sympy.NumberSymbol, sym_theta: sympy.NumberSymbol) -> None:
        self.sym_alpha = sym_alpha
        self.sym_a = sym_a
        self.sym_d = sym_d
        self.sym_theta = sym_theta
    
    def transf(self) -> numpy.ndarray:
        return DH_transformation(self.alpha, self.a, self.d, self.theta)
    
    def sym_transf(self) -> sympy.Matrix:
        return sym_DH_transformation(self.sym_alpha, self.sym_a, self.sym_d, self.sym_theta)

def DH_revolute_joint(alpha: float, a: float,
                      d: float, theta: float,
                      sym_alpha: sympy.NumberSymbol, sym_a: sympy.NumberSymbol,
                      sym_d: sympy.NumberSymbol, sym_theta: sympy.NumberSymbol) -> DH_Joint:
    return DH_Joint("Revolute", alpha, a, d, theta, sym_alpha, sym_a, sym_d, sym_theta)

def DH_prismatic_joint(alpha: float, a: float,
                      d: float, theta: float,
                      sym_alpha: sympy.NumberSymbol, sym_a: sympy.NumberSymbol,
                      sym_d: sympy.NumberSymbol, sym_theta: sympy.NumberSymbol) -> DH_Joint:
    return DH_Joint("Prismatic", alpha, a, d, theta, sym_alpha, sym_a, sym_d, sym_theta)

class DH_Robot():
    def __init__(self, joints: Iterable[DH_Joint]) -> None:
        self.joints = joints
    
    def set_alpha(self, values: Iterable[float]) -> None:
        if (len(values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            alpha: float = values[i]
            joint.alpha = alpha
        
    def set_sym_alpha(self, sym_values: Iterable[sympy.NumberSymbol]) -> None:
        if (len(sym_values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(sym_values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            sym_alpha: sympy.NumberSymbol = sym_values[i]
            joint.sym_alpha = sym_alpha
    
    def set_a(self, values: Iterable[float]) -> None:
        if (len(values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            a: float = values[i]
            joint.a = a
        
    def set_sym_a(self, sym_values: Iterable[sympy.NumberSymbol]) -> None:
        if (len(sym_values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(sym_values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            sym_a: sympy.NumberSymbol = sym_values[i]
            joint.sym_a = sym_a
    
    def set_d(self, values: Iterable[float]) -> None:
        if (len(values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            d: float = values[i]
            joint.d = d
        
    def set_sym_d(self, sym_values: Iterable[sympy.NumberSymbol]) -> None:
        if (len(sym_values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(sym_values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            sym_d: sympy.NumberSymbol = sym_values[i]
            joint.sym_d = sym_d
    
    def set_theta(self, values: Iterable[float]) -> None:
        if (len(values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            theta: float = values[i]
            joint.theta = theta
        
    def set_sym_theta(self, sym_values: Iterable[sympy.NumberSymbol]) -> None:
        if (len(sym_values) != len(self.joints)):
            raise ValueError(f"Wrong number of parameters: expected {len(self.joints)}, got {len(sym_values)}")
        for i in range(len(self.joints)):
            joint: DH_Joint = self.joints[i]
            sym_theta: sympy.NumberSymbol = sym_values[i]
            joint.sym_theta = sym_theta
    
    def total_transf(self) -> numpy.ndarray:
        transfm = numpy.eye(4)
        for joint in self.joints:
            transfm = transfm @ joint.transf()
        return transfm
    
    def sym_total_transf(self) -> sympy.NumberSymbol:
        transfm = sympy.Identity(4)
        for joint in self.joints:
            transfm *= joint.sym_transf()
        return sympy.simplify(transfm)
