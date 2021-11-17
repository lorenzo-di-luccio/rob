import sympy
from typing import Iterable

def sym_vec(v: Iterable[sympy.NumberSymbol]) -> sympy.Matrix:
    return sympy.Matrix([v])

def sym_colvec(v: Iterable[sympy.NumberSymbol]) -> sympy.Matrix:
    return sympy.Matrix(v)

def sym_unitvec(v: Iterable[sympy.NumberSymbol]) -> sympy.Matrix:
    vec = sympy.Matrix([v])
    return sympy.simplify(vec.normalized())

def sym_colunitvec(v: Iterable[float]) -> sympy.Matrix:
    vec = sympy.Matrix(v)
    return sympy.simplify(vec.normalized())

def sym_isunitvec(vec: sympy.Matrix) -> bool:
    return vec.norm() == 1

def sym_iszerovec(vec: sympy.Matrix) -> bool:
    return vec.norm() == 0
