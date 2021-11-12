import math
import numpy
import sympy
import sympy.printing
from typing import *

PI: float = numpy.pi
SYM_PI: sympy.Symbol = sympy.pi
EPS: float = 1.e-8
INFINITY: float = numpy.inf

def sym(name: str) -> sympy.Symbol:
    return sympy.Symbol(name, real=True)

def matvec_print(expr: numpy.ndarray) -> None:
    print(expr)

def sym_print(expr: sympy.Expr) -> None:
    sympy.printing.pretty_print(expr)

def rad(angle: float) -> float:
    return angle * PI / 180.0

def sym_rad(angle: sympy.NumberSymbol) -> sympy.NumberSymbol:
    return sympy.simplify(angle * SYM_PI / 180)

def deg(angle: float) -> float:
    return angle * 180.0 / PI

def sym_deg(angle: sympy.NumberSymbol) -> float:
    return sympy.simplify(angle * 180 / SYM_PI)

def abs(x: float) -> float:
    return x if x >= 0.0 else -x

def sin(x: float) -> float:
    s = math.sin(x)
    return 0.0 if abs(s) < EPS else s

def cos(x: float) -> float:
    c = math.cos(x)
    return 0.0 if abs(c) < EPS else c
