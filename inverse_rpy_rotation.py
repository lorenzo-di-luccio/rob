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
sol = rob.inverse_rot_rpy(r)
print("\n\n")
if (sol["singular"]):
    print("SINGULAR CASE")
    sol_ = sol["solution"][0]
    if (abs(sol_[1] - rob.PI / 2.) < rob.EPS):
        print("Infinite solutions for yaw - roll:")
        print("pitch in deg:")
        rob.matvec_print(rob.deg(sol_[1]))
        print("pitch in rad:")
        rob.matvec_print(sol_[1])
        print("yaw - roll in deg:")
        rob.matvec_print(rob.deg(sol["op"]))
        print("yaw - roll in rad:")
        rob.matvec_print(sol["op"])
    else:
        print("Infinite solutions for yaw + roll:")
        print("pitch in deg:")
        rob.matvec_print(rob.deg(sol_[1]))
        print("pitch in rad:")
        rob.matvec_print(sol_[1])
        print("yaw + roll in deg:")
        rob.matvec_print(rob.deg(sol["op"]))
        print("yaw + roll in rad:")
        rob.matvec_print(sol["op"])
else:
    print("REGULAR CASE")
    print("Solution 1:")
    sol1 = sol["solution"][0]
    print("roll in deg:")
    rob.matvec_print(rob.deg(sol1[0]))
    print("roll in rad:")
    rob.matvec_print(sol1[0])
    print("pitch in deg:")
    rob.matvec_print(rob.deg(sol1[1]))
    print("pitch in rad:")
    rob.matvec_print(sol1[1])
    print("yaw in deg:")
    rob.matvec_print(rob.deg(sol1[2]))
    print("yaw in rad:")
    rob.matvec_print(sol1[2])
    print(25 * "*")
    print("Solution 2:")
    sol2 = sol["solution"][1]
    print("roll in deg:")
    rob.matvec_print(rob.deg(sol2[0]))
    print("roll in rad:")
    rob.matvec_print(sol2[0])
    print("pitch in deg:")
    rob.matvec_print(rob.deg(sol2[1]))
    print("pitch in rad:")
    rob.matvec_print(sol2[1])
    print("yaw in deg:")
    rob.matvec_print(rob.deg(sol2[2]))
    print("yaw in rad:")
    rob.matvec_print(sol2[2])
