import numpy

EPS: float = numpy.finfo(numpy.float64).eps
PI: float = numpy.pi

def num_print(obj) -> None:
    with numpy.printoptions(precision=8) as popts:
        print(obj)

def rad(angle: float) -> float:
    return angle * PI / 180.

def deg(angle: float) -> float:
    return angle * 180. / PI

def wrap_0_2pi(angle: float) -> float:
    return angle - 2. * PI * numpy.floor(angle / 2. / PI)

def wrap_mpi_pi(angle: float) -> float:
    return numpy.mod(angle + PI, 2. * PI) - PI
