import rob
import numpy
import sympy

# NUMERIC
roll = 45.
pitch = 0.
yaw = 0.
print("Roll in deg:")
rob.matvec_print(roll)
roll = rob.rad(roll)
print("Roll in rad:")
rob.matvec_print(roll)
print("Pitch in deg:")
rob.matvec_print(pitch)
pitch = rob.rad(pitch)
print("Pitch in rad:")
rob.matvec_print(pitch)
print("Yaw in deg:")
rob.matvec_print(yaw)
yaw = rob.rad(yaw)
print("Yaw in rad:")
rob.matvec_print(yaw)
r = rob.rot_rpy(roll, pitch, yaw)
print("\n\n")
print("Solution:")
print("rotation matrix:")
rob.matvec_print(r)

# SYMBOLIC
'''roll = 45
pitch = 0
yaw = 0
print("Roll in deg:")
rob.sym_print(roll)
roll = rob.rad(roll)
print("Roll in rad:")
rob.sym_print(roll)
print("Pitch in deg:")
rob.sym_print(pitch)
pitch = rob.rad(pitch)
print("Pitch in rad:")
rob.sym_print(pitch)
print("Yaw in deg:")
rob.sym_print(yaw)
yaw = rob.rad(yaw)
print("Yaw in rad:")
rob.sym_print(yaw)
r = rob.sym_rot_rpy(roll, pitch, yaw)
print("\n\n")
print("Solution:")
print("rotation matrix:")
rob.sym_print(r)'''
