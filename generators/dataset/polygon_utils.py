from collections import defaultdict
from typing import Iterable


from generators.dataset.my_types import Edge, ShapelyPolygon, ShapelyPoint


def buildPolygonFromEdges(edges: Iterable[Edge]) -> ShapelyPolygon:
    """
    Take an edge list and construct a shapely Polygon

    This has a special construction,
    It is a list of points, in order, which define the polygon
    Then there is a list of additional points that define holes in the polygon
    """

    # a mapping of points to adjacent points
    edge_index = defaultdict(list)

    point_set = set()

    for p0, p1 in edges:
        edge_index[p0].append(p1)
        edge_index[p1].append(p0)

        # can't just call buildPointSet cause 
        #   edges might be a single-use iterator
        point_set.add(p0)
        point_set.add(p1)

    used_points = set()

    all_poly = [] # list of all polygons
    b_box = [] # list of polygon bound boxes

    for p in point_set:
        if p in used_points:
            continue
        
        this_poly = [p]
        min_x, min_y = p
        max_x, max_y = p
        used_points.add(p)

        while True:
            m = edge_index[p]

            added = False

            for p_new in m:
                if p_new in used_points:
                    continue
                this_poly.append(p_new)
                used_points.add(p_new)
                p = p_new
                added = True

                min_x = min(min_x, p_new[0])
                max_x = max(max_x, p_new[0])
                min_y = min(min_y, p_new[1])
                max_y = max(max_y, p_new[1])
                break

            if not added:
                # we have used all the points
                break

        all_poly.append(this_poly)
        b_box.append((min_x, min_y, max_x, max_y))

    # now find the one that is the outside polygon

    biggest_idx = 0
    biggest_bbox = b_box[0]

    for i, (_, b) in enumerate(zip(all_poly, b_box)):
        if i == 0:
            continue

        if (b[0] < biggest_bbox[0]
            or b[1] < biggest_bbox[1]
            or b[2] > biggest_bbox[2] 
            or b[3] > biggest_bbox[3]):
            # this is the new biggest bounding box
            biggest_idx = i
            biggest_bbox = b
    
    interiors = [f for i,f in enumerate(all_poly) if i != biggest_idx]

    return ShapelyPolygon(all_poly[biggest_idx], interiors)

