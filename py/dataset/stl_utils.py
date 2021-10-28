

from collections import defaultdict
from stl import mesh, Mode
from typing import Callable, Iterable, Optional, Tuple, Union
from py.dataset.env import ROUND_PRECISION

from py.dataset.vector_utils import forcePositiveVector, vectorDot
from py.dataset.my_types import EDGE_SELECTOR_NONE, Edge, EdgeList, Point, PointSet, Vector, EdgeDict

def loadSTLData(filepath: str):
    """Loads the file at filepath with stl library"""
    return mesh.Mesh.from_file(filepath, mode=Mode.BINARY)


def determineIfLaserNormal(stl_data) -> Optional[Vector]:
    """
    Iterate the stl data and determine if the part has a laser normal
    Laser normal is defined as:
        1. Two normals that are opposite directions
        2. These normals are perpendicular to all other normals
    """

    # set which will store all the normals made positive
    s = set()

    # iterate over all the normals 
    for unit_normal in stl_data.get_unit_normals():
        un = forcePositiveVector(unit_normal)

        # TODO - there is potential for optimization here:
        #   can short-circut as soon as there is three non-perpendicular normals
        s.add(tuple(map(lambda x: round(x, ROUND_PRECISION), un)))

    for n in s:
        valid = True

        for t in s:
            if t == n:
                continue

            d = vectorDot(n, t)

            if d != 0:
                # these vectors are not perpendicular
                #   and are not the same
                #   so this cannot be a laser normal
                valid = False
                break

        if valid:
            return n
    
    return None


def projectPointsEdgesOntoPlane(
    stl_data, 
    projection_normal: Vector, 
    keep_planar: bool = False, 
    keep_non_planar: bool = False
) -> Tuple[PointSet, EdgeDict]:
    """
    Project the points and edges of the stl data onto 
        a plane with the provided normal through origin 
    @param keep_planar controls whether or not to project items already in a parallel plane
        this is more of a filter than an actual operation; but the projection will still occur
    @param keep_non_planar controls whether or not to project items not in the plane
    One of keep_planar and keep_non_planar should be true, or the result will be empty
    Returns a tuple
        index 0 - list of all points
        index 1 - a defaultdict(int) of each edge and number of occurrences
    """

    projected_edges : EdgeDict = defaultdict(int)
    projected_points : PointSet = set()

    if not (keep_non_planar or keep_planar):
        # both are empty, nothing is kept, short circuit
        return (projected_points, projected_edges)

    for points, norm in zip(stl_data.points, stl_data.get_unit_normals()):
        if (norm == projection_normal).all():
            # this is a parallel plane
            if not keep_planar:
                continue
        else:
            # this is not in a parallel plane
            #   notice that if projection_normal is a laser normal, this 
            #       plane will be perpendicular and thus project 
            #       into a line
            if not keep_non_planar:
                continue

        assert(len(points) == 9)
        pts = (points[:3], points[3:6], points[6:])

        proj_pts = []

        for p in pts:
            d = vectorDot(p, projection_normal)
            p_proj = tuple(map(lambda x,y: round(x - d*y, ROUND_PRECISION), p, projection_normal))
            if p_proj not in proj_pts:
                proj_pts.append(p_proj)
                projected_points.add(p_proj)
        
        for i in range(len(proj_pts)):
            for j in range(i+1, len(proj_pts)):
                m = proj_pts[i]
                n = proj_pts[j]

                assert(len(n) == 3)
                assert(len(m) == 3)

                if n == m:
                    # should be unreachable
                    continue
                elif n < m:
                    t = m
                    m = n
                    n = t
                projected_edges[(m,n)] += 1
    return (projected_points, projected_edges)


def transformPoints(
    points: Iterable[Point],
    transform: Callable[[Point], PointSet],
    edges: Optional[EdgeDict] = None
) -> Union[PointSet, Tuple[PointSet, EdgeDict]]:
    """
    Apply the transform to all points
    Optionally, provide an edge dict and apply the transform to that too
    Transform may change dimensionality of the data
    Returns either a PointSet in the case that edges = None
        otherwise returns a tuple of PointSet and EdgeDict 
    """

    new_points = map(transform, points)

    if edges is None:
        return new_points

    new_edges = defaultdict(int)

    for (p1, p2), v in edges.items():
        kn = (transform(p1), transform(p2))
        new_edges[kn] = v
    
    return (new_points, new_edges)


def buildPointSetFromEdges(edges: Iterable[Edge]) -> PointSet:
    """Builds the point set from the edges"""
    ps = set()

    for p1, p2 in edges:
        ps.add(p1)
        ps.add(p2)

    return ps


def doEdgeReduction(
    edges: EdgeDict,
    selector: Callable[[Point, Point, int], bool],
) -> EdgeList:
    """
    Apply `selector` to all `edges`
    @param selector
        A callable that takes the end points of the edge and the number of times
        the edge appears and returns a bool indicating if this edge should be kept

        Default value picks exclusively the unique edges
    Use buildPointSetFromEdges to get the new point set
    """
    el = []

    for (p1, p2), v in edges.items():
        if selector(p1, p2, v):
            el.append((p1, p2))
    return el
