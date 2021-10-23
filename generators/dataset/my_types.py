from typing import Tuple, DefaultDict, Set

Point = Tuple[float, ...]
Vector = Tuple[float, ...]

Edge = Tuple[Point, Point]
PointSet = Set[Point]
EdgeDict = DefaultDict[Edge, int]
