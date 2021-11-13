import dataclasses
import numpy

from .base import wrap_mpi_pi
from .transforms2d import isrot2, ishom2

@dataclasses.dataclass
class InverseRot2Solution():
    is_ok: bool
    msg: str
    num_solutions: int
    angle: float

@dataclasses.dataclass
class InverseHomRot2Solution():
    is_ok: bool
    msg: str
    num_solutions: int
    angle: float

@dataclasses.dataclass
class InverseTransl2Solution():
    is_ok: bool
    msg: str
    num_solutions: int
    tx: float
    ty: float

@dataclasses.dataclass
class InverseHomTransl2Solution():
    is_ok: bool
    msg: str
    num_solutions: int
    tx: float
    ty: float

@dataclasses.dataclass
class InverseHom2Solution():
    is_ok: bool
    msg: str
    num_solutions: int
    angle: float
    tx: float
    ty: float

def inverse_rot2(rotmat: numpy.ndarray) -> InverseRot2Solution:
    if not isrot2(rotmat):
        return InverseRot2Solution(False, "Not a rot2 matrix", 0, None)
    return InverseRot2Solution(True, None, 1, wrap_mpi_pi(numpy.arctan2(rotmat[1, 0], rotmat[0, 0])))

def inverse_hom_rot2(homrotmat: numpy.ndarray) -> InverseHomRot2Solution:
    if not ishom2(homrotmat):
        return InverseHomRot2Solution(False, "Not a hom_rot2 matrix", 0, None)
    return InverseHomRot2Solution(True, None, 1, wrap_mpi_pi(numpy.arctan2(homrotmat[1, 0], homrotmat[0, 0])))

def inverse_rot2(translvec: numpy.ndarray) -> InverseTransl2Solution:
    translvec = translvec.reshape((-1,))
    return InverseTransl2Solution(True, None, 1, translvec[0], translvec[1])

def inverse_hom_transl2(homtranslmat: numpy.ndarray) -> InverseHomTransl2Solution:
    if not ishom2(homtranslmat):
        return InverseHomTransl2Solution(False, "Not a hom_transl2 matrix", 0, None, None)
    return InverseHomTransl2Solution(True, None, 1, homtranslmat[0, 2], homtranslmat[1, 2])

def inverse_hom2(hommat: numpy.ndarray) -> InverseHom2Solution:
    if not ishom2(hommat):
        return InverseHom2Solution(False, "Not a hom2 matrix", 0, None, None, None)
    return InverseHom2Solution(True, None, 1, wrap_mpi_pi(numpy.arctan2(hommat[1, 0], hommat[0, 0])),
                               hommat[0, 2], hommat[1, 2])
