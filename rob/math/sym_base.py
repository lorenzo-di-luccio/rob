import sympy
import sympy.printing

SYM_PI: sympy.Symbol = sympy.pi

def sym(name: str) -> sympy.Symbol:
    return sympy.Symbol(name, real=True)

def sym_vec(name: str, n: int) -> sympy.MatrixSymbol:
    return sympy.MatrixSymbol(name, 1, n)

def sym_colvec(name: str, n: int) -> sympy.MatrixSymbol:
    return sympy.MatrixSymbol(name, n, 1)

def sym_mat(name: str, m: int, n: int) -> sympy.MatrixSymbol:
    return sympy.MatrixSymbol(name, m, n)

def sym_print(obj) -> None:
    sympy.printing.pretty_print(obj)

def sym_rad(angle: sympy.NumberSymbol) -> sympy.NumberSymbol:
    return sympy.simplify(angle * SYM_PI / 180)

def sym_deg(angle: sympy.NumberSymbol) -> sympy.NumberSymbol:
    return sympy.simplify(angle * 180 / SYM_PI)

def sym_wrap_0_2pi(angle: sympy.NumberSymbol) -> sympy.NumberSymbol:
    return sympy.simplify(angle - 2 * SYM_PI * sympy.floor(angle / 2 / SYM_PI))

def sym_wrap_mpi_pi(angle: sympy.NumberSymbol) -> sympy.NumberSymbol:
    return sympy.simplify(sympy.Mod(angle + SYM_PI, 2 * SYM_PI) - SYM_PI)
