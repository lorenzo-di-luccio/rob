import math
import numpy
from typing import *
from .robotics_math import EPS, PI, abs

def inverse_rot(rot: numpy.ndarray) -> dict:
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

def inverse_rot_rpy(rot: numpy.ndarray) -> dict:
    s_angley = -rot[2, 0]
    c1_angley = math.sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2])
    c2_angley = -c1_angley
    angley1 = math.atan2(s_angley, c1_angley)
    if (abs(angley1 - PI / 2.) < EPS):
        # phi - psi
        c = rot[1, 1]
        s = rot[1, 2]
        angle_z_minus_x = math.atan2(s, c)
        return {
            "singular": True,
            "solution": [
                (None, PI / 2., None)
            ],
            "op": angle_z_minus_x
        }
    if (abs(angley1 + PI / 2.) < EPS):
        #phi + psi
        c = rot[1, 1]
        s = -rot[1, 2]
        angle_z_plus_x = math.atan2(s, c)
        return {
            "singular": True,
            "solution": [
                (None, -PI / 2., None)
            ],
            "op": angle_z_plus_x
        }
    angley2 = math.atan2(s_angley, c2_angley)
    s1_anglex = rot[2, 1] / c1_angley
    s2_anglex = rot[2, 1] / c2_angley
    c1_anglex = rot[2, 2] / c1_angley
    c2_anglex = rot[2, 2] / c2_angley
    anglex1 = math.atan2(s1_anglex, c1_anglex)
    anglex2 = math.atan2(s2_anglex, c2_anglex)
    s1_anglez = rot[1, 0] / c1_angley
    s2_anglez = rot[1, 0] / c2_angley
    c1_anglez = rot[0, 0] / c1_angley
    c2_anglez = rot[0, 0] / c2_angley
    anglez1 = math.atan2(s1_anglez, c1_anglez)
    anglez2 = math.atan2(s2_anglez, c2_anglez)
    return {
        "singular": False,
        "solution": [
            (anglex1, angley1, anglez1),
            (anglex2, angley2, anglez2)
        ]
    }