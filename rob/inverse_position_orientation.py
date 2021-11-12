import math
import numpy
from typing import *
from .robotics_math import EPS, PI, abs

def inverse_rot(rot: numpy.ndarray):
    r1 = rot[0, 1] - rot[1, 0]
    r2 = rot[0, 2] - rot[2, 0]
    r3 = rot[1, 2] - rot[2, 1]
    s = math.sqrt(r1 * r1 + r2 * r2 + r3 * r3) / 2.
    c = (rot[0, 0] + rot[1, 1] + rot[2, 2] - 1.) / 2.
    angle1 = math.atan2(s, c)
    if (abs(angle1) < EPS):
        return {
            "singular": True,
            "solution": [
                (None, angle1)
            ]
        }
    if (abs(angle1 - PI) < EPS or abs(angle1 + PI) < EPS):
        axis1 = numpy.array([
            math.sqrt((rot[0, 0] + 1) / 2.),
            math.sqrt((rot[1, 1] + 1) / 2.),
            math.sqrt((rot[2, 2] + 1) / 2.)
        ], dtype=float)
        return {
            "singular": True,
            "solution": [
                (axis1, angle1)
            ]
        }
    angle2 = math.atan2(-s, c)
    axis = numpy.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]], dtype=float)
    axis1 = axis * (1. / (2. * math.sin(angle1)))
    axis2 = axis * (1. / (2. * math.sin(angle2)))
    return {
        "singular": False,
        "solution": [
            (axis1, angle1),
            (axis2, angle2)
        ]
    }
