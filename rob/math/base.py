import numpy

EPS: float = 1.e-8
PI: float = numpy.pi

def num_print(obj) -> None:
    with numpy.printoptions(precision=8, suppress=True) as popts:
        print(obj)

def rad(angle: float) -> numpy.float64:
    return angle * PI / 180.

def deg(angle: float) -> numpy.float64:
    return angle * 180. / PI

def wrap_0_2pi(angle: float) -> numpy.float64:
    return angle - 2. * PI * numpy.floor(angle / 2. / PI, dtype=numpy.float64)

def wrap_mpi_pi(angle: float) -> numpy.float64:
    return numpy.mod(angle + PI, 2. * PI, dtype=numpy.float64) - PI

def abs(x: float) -> numpy.float64:
    return numpy.abs(x, dtype=numpy.float64)
