import numpy
import sympy
import sympy.printing
from typing import *

PI: float = numpy.pi
SYM_PI: sympy.Symbol = sympy.pi

def sym(name: str) -> sympy.Symbol:
    return sympy.Symbol(name, real=True)

def sym_print(expr: sympy.Expr) -> None:
    sympy.printing.pretty_print(expr)

def rad(angle: float) -> float:
    return angle * PI / 180.0

def sym_rad(angle: sympy.NumberSymbol) -> sympy.NumberSymbol:
    return angle * SYM_PI / 180

def deg(angle: float) -> float:
    return angle * 180.0 / PI

def sym_deg(angle: sympy.NumberSymbol) -> float:
    return angle * 180 / SYM_PI
