import rob
import numpy
import sympy

# NUMERIC
axis = numpy.array([1, 0, 0], dtype=float)
angle = 45.
print("Axis:")
rob.matvec_print(axis)
print("Angle in deg:")
rob.matvec_print(angle)
angle = rob.rad(angle)
print("Angle in rad:")
rob.matvec_print(angle)
r = rob.rot(axis, angle)
print("\n\n")
print("Solution:")
print("rotation matrix:")
rob.matvec_print(r)

# SYMBOLIC
'''axis = sympy.Matrix([1, 0, 0])
angle = rob.sym("theta")
print("Axis:")
rob.sym_print(axis)
print("Angle in deg:")
rob.sym_print(rob.sym_deg(angle))
print("Angle in rad:")
rob.sym_print(angle)
r = rob.sym_rot(axis, angle)
print("\n\n")
print("Solution:")
print("rotation matrix:")
rob.sym_print(r)'''