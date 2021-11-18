import dataclasses
import numpy
from typing import List

from .base import abs, wrap_mpi_pi, PI, EPS
from .transforms3d import isrot, ishom

@dataclasses.dataclass
class InverseRotSolution():
    is_ok: bool
    msg: str
    is_singular: bool
    num_solutions: int
    axes: List[numpy.ndarray]
    angles: List[float]
    axisx_times_axisy: float
    axisx_times_axisz: float
    axisy_times_axisz: float

@dataclasses.dataclass
class InverseRotRPYXYZSolution():
    is_ok: bool
    msg: str
    is_singular: bool
    num_solutions: int
    anglesx: List[float]
    anglesy: List[float]
    anglesz: List[float]
    anglez_plus_anglex: float
    anglez_minus_anglex: float

def inverse_rot(rotmat: numpy.ndarray) -> InverseRotSolution:
    if not isrot(rotmat):
        return InverseRotSolution(False, "Not a rot matrix", None, 0, None, None, None, None, None)
    r1 = rotmat[0, 1] - rotmat[1, 0]
    r2 = rotmat[0, 2] - rotmat[2, 0]
    r3 = rotmat[1, 2] - rotmat[2, 1]
    scaled_s = numpy.sqrt(r1 * r1 + r2 * r2 + r3 * r3, dtype=numpy.float64)
    scaled_c = rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2] - 1.
    angle1 = wrap_mpi_pi(numpy.arctan2(scaled_s, scaled_c, dtype=numpy.float64))
    if abs(angle1) < EPS:
        return InverseRotSolution(True, "sin(angle)=0 & angle=0", True, -1, [], [0.], None, None, None)
    if abs(angle1 - PI) < EPS or abs(angle1 + PI) < EPS:
        axis = numpy.array([
            numpy.sqrt((rotmat[0, 0] + 1.) / 2., dtype=numpy.float64),
            numpy.sqrt((rotmat[1, 1] + 1.) / 2., dtype=numpy.float64),
            numpy.sqrt((rotmat[2, 2] + 1.) / 2., dtype=numpy.float64)
        ], dtype=numpy.float64)
        return InverseRotSolution(True, "sin(angle)=0 & angle=+-pi", True, 27, [axis], [PI],
                                  rotmat[0, 1] / 2., rotmat[0, 2] / 2., rotmat[1, 2] / 2.)
    s = scaled_s / 2.
    angle2 = numpy.arctan2(-scaled_s, scaled_c, dtype=numpy.float64)
    scaled_axis = numpy.array([
        rotmat[2, 1] - rotmat[1, 2], rotmat[0, 2] - rotmat[2, 0], rotmat[1, 0] - rotmat[0, 1]
    ], dtype=numpy.float64) / 2.
    axis1 = scaled_axis / s
    axis2 = scaled_axis / (-s)
    return InverseRotSolution(True, None, False, 2, [axis1, axis2], [angle1, angle2], None, None, None)

def inverse_rot_rpyxyz(rotmat: numpy.ndarray) -> InverseRotRPYXYZSolution:
    if not isrot(rotmat):
        return InverseRotRPYXYZSolution(False, "Not a rot matrix", None, 0, None, None, None, None, None)
    sy = -rotmat[2, 0]
    cy1 = numpy.sqrt(rotmat[2, 1] * rotmat[2, 1] + rotmat[2, 2] * rotmat[2, 2], dtype=numpy.float64)
    angle1y = wrap_mpi_pi(numpy.arctan2(sy, cy1, dtype=numpy.float64))
    if abs(angle1y - PI / 2.) < EPS:
        c = rotmat[1, 1]
        s = rotmat[1, 2]
        anglez_minus_anglex = numpy.arctan2(s, c, dtype=numpy.float64)
        return InverseRotRPYXYZSolution(True, "cos(angley)=pi/2", True, -1, [], [PI / 2.], [],
                                        None, anglez_minus_anglex)
    if abs(angle1y + PI / 2.) < EPS:
        c = rotmat[1, 1]
        s = -rotmat[1, 2]
        anglez_plus_anglex = numpy.arctan2(s, c, dtype=numpy.float64)
        return InverseRotRPYXYZSolution(True, "cos(angley)=-pi/2", True, -1, [], [-PI / 2.], [],
                                        anglez_plus_anglex, None)
    cy2 = -cy1
    angle2y = wrap_mpi_pi(numpy.arctan2(sy, cy2, dtype=numpy.float64))
    sx1 = rotmat[2, 1] / cy1
    sx2 = rotmat[2, 1] / cy2
    cx1 = rotmat[2, 2] / cy1
    cx2 = rotmat[2, 2] / cy2
    sz1 = rotmat[1, 0] / cy1
    sz2 = rotmat[1, 0] / cy2
    cz1 = rotmat[0, 0] / cy1
    cz2 = rotmat[0, 0] / cy2
    angle1x = wrap_mpi_pi(numpy.arctan2(sx1, cx1, dtype=numpy.float64))
    angle2x = wrap_mpi_pi(numpy.arctan2(sx2, cx2, dtype=numpy.float64))
    angle1z = wrap_mpi_pi(numpy.arctan2(sz1, cz1, dtype=numpy.float64))
    angle2z = wrap_mpi_pi(numpy.arctan2(sz2, cz2, dtype=numpy.float64))
    return InverseRotRPYXYZSolution(True, None, False, 2,
                                    [angle1x, angle2x], [angle1y, angle2y], [angle1z, angle2z], None, None)
