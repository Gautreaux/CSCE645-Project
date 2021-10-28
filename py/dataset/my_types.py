from typing import List, Optional, Tuple, DefaultDict, Set

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint

Point = Tuple[float, ...]
Vector = Tuple[float, ...]

Edge = Tuple[Point, Point]
PointSet = Set[Point]
EdgeDict = DefaultDict[Edge, int]

PointList = List[Point]
EdgeList = List[Edge]


# Common patterns for various functions

## Dimension Reduction transforms for transformPoints

# keep only the 1st and 2nd dimension
TRANSFORM_KEEP_XY = (lambda x: (x[0], x[1]))

# keep only the 1st and 3rd dimension
TRANSFORM_KEEP_XZ = (lambda x: (x[0], x[2]))

# keep only the 2nd and 3rd dimension
TRANSFORM_KEEP_YZ = (lambda x: (x[1], x[2]))

## Edge selectors for buildPointSetFromEdges:

# select none of the edges
EDGE_SELECTOR_NONE = (lambda p1, p2, q: False)

# select all of the edges
EDGE_SELECTOR_ALL = (lambda p1, p2, q: True)

# select the unique edges (edges that appear exactly once)
EDGE_SELECTOR_UNIQUE = (lambda p1, p2, q: q == 1)