import dataclasses
import numpy
import numpy.linalg

from .base import wrap_mpi_pi
from .vectors_matrices import isrot2, ishom2

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

def inverse_rot2(rotmat: numpy.ndarray) -> InverseRot2Solution:
    if not isrot2(rotmat):
        return InverseRot2Solution(False, "Not a rot2 matrix", 0, None)
    return InverseRot2Solution(True, None, 1, wrap_mpi_pi(numpy.arctan2(rotmat[1, 0], rotmat[0, 0])))

def inverse_hom_rot2(homrotmat: numpy.ndarray) -> InverseRot2Solution:
    if not ishom2(homrotmat):
        return InverseRot2Solution(False, "Not a hom_rot2 matrix", 0, None)
    return InverseHomRot2Solution(True, None, 1, wrap_mpi_pi(numpy.arctan2(homrotmat[1, 0], homrotmat[0, 0])))
