

from py.dataset.env import ROUND_PRECISION

from py.dataset.my_types import Vector


def forcePositiveVector(v: Vector) -> Vector:
    """
    Force v to be a positive vector:
    the first non-zero term is positive
    """
    
    for t in v:
        if t > 0:
            return v
        if t < 0:
            return tuple(map(lambda x: -x, v))

    # this is a zero vector
    return v


def vectorDot(v: Vector, vv: Vector) -> float:
    """Compute the dot product of the vectors"""
    return round(sum(map(lambda x,y:x*y, v, vv)), ROUND_PRECISION)