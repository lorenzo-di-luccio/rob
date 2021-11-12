import rob
import math
import numpy

r = numpy.array([
    [-1, 0, 0],
    [0, -1 / math.sqrt(2), -1 / math.sqrt(2)],
    [0, -1 / math.sqrt(2), 1 / math.sqrt(2)]
], dtype=float)
print("Rotation matrix:")
rob.matvec_print(r)
sol = rob.inverse_rot(r)
print("\n\n")
if (sol["singular"]):
    print("SINGULAR CASE")
    sol_ = sol["solution"][0]
    if (sol_[0] is None):
        print("Infinite solutions for axis:")
        print("axis:")
        print(sol_[0])
        print("angle in deg:")
        rob.matvec_print(rob.deg(sol_[1]))
        print("angle in rad:")
        rob.matvec_print(sol_[1])
    else:
        print("Multiple solutions, choose well the signs:")
        print("axis:")
        rob.matvec_print(sol_[0])
        print("angle in deg:")
        rob.matvec_print(rob.deg(sol_[1]))
        print("angle in rad:")
        rob.matvec_print(sol_[1])
else:
    print("REGULAR CASE")
    print("Solution 1:")
    sol1 = sol["solution"][0]
    print("axis:")
    rob.matvec_print(sol1[0])
    print("angle in deg:")
    rob.matvec_print(rob.deg(sol1[1]))
    print("angle in rad:")
    rob.matvec_print(sol1[1])
    print(25 * "*")
    print("Solution 2:")
    sol2 = sol["solution"][1]
    print("axis:")
    rob.matvec_print(sol2[0])
    print("angle in deg:")
    rob.matvec_print(rob.deg(sol2[1]))
    print("angle in rad:")
    rob.matvec_print(sol2[1])
    print(10 * "*")