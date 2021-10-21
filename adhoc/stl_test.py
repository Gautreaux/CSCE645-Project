
from collections import defaultdict
from typing import Counter
import itertools
from numpy import ceil, floor, int_
import random
from stl import mesh, Mode
from matplotlib import pyplot as plt

LASER_CUT_PART = "basebot.stl"
NON_LASER_PART = "omniwheel.stl"

IDEAL_NORMAL = (0, 1, 0)

laser_mesh = mesh.Mesh.from_file(LASER_CUT_PART, mode=Mode.BINARY)

# for p in laser_mesh.normals:

s = set()

for p_unit in laser_mesh.get_unit_normals():
    # mag = sum(map(lambda x: x**2, p))**.5
    # p_unit = tuple(map(lambda x: round(x / mag, 6), p))
    # # print(p_unit)

    # force the vector to be positive
    mod = 1
    if p_unit[0] == 0:
        if p_unit[1] == 0:
            if p_unit[2] < 0:
                mod = -1
        elif p_unit[1] < 0:
            mod = -1
    elif p_unit[0] < 0:
        mod = -1

    # we could probably highly optimize by checking normal in here
    #  would need to maintain a set of candidate normals and check against that on all inserts
    s.add(tuple(map(lambda x: round(x*mod, 6), p_unit)))

print(f"Considering {len(s)} unique normals")

for normal in s:
    valid = True
    
    for test_normal in s:
        if test_normal == normal:
            continue

        dot_test = round(sum(map(lambda x,y: x * y, normal, test_normal)), 6)

        if dot_test == 0:
            # these are a right angle:
            pass
        else:
            # these vectors do not form a right angle, this cannot be a laser cut axis
            valid = None
            break
    
    if valid:
        valid = normal
        print(f"Found a laser cut normal with {normal}")
        break

if not valid:
    print(f"No laser cut normal found")
    exit(1)
    assert(0)

# now lets reduce to 2d points:
laser_normal = valid

plane_pts = []

in_plane_edges = defaultdict(int)

# for all the points, project onto the plane
#   with normal found earlier and through origin
for p, norm in zip(laser_mesh.points, laser_mesh.get_unit_normals()):
    if not (norm == laser_normal).all():
        continue

    x1, y1, z1, x2, y2, z2, x3, y3, z3 = p
    p_tuples = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
    plane_pts.extend(p_tuples)

    s = set()
    s.add((p_tuples[0], p_tuples[1]))
    s.add((p_tuples[1], p_tuples[2]))
    s.add((p_tuples[0], p_tuples[2]))

    assert(len(s) == 3)

    for m,n in s:
        if m == n:
            continue
        if n < m:
            t = m
            m = n
            n = t
        assert(m < n)

        in_plane_edges[(m,n)] += 1


relevant_edges = {k for k,v in in_plane_edges.items() if v == 1}

relevant_points = set()
for edge in relevant_edges:
    relevant_points.add(edge[0])
    relevant_points.add(edge[1])

# rotate the 3d points so that the laser normal aligns to the ideal normal
# TODO
assert(laser_normal == IDEAL_NORMAL)

# plot the points for sanity
plt.scatter(list(map(lambda x: x[0], relevant_points)), list(map(lambda x: x[2], relevant_points)))

for edge in relevant_edges:
    plt.plot([edge[0][0], edge[1][0]],[edge[0][2], edge[1][2]])
plt.axis('equal')
# plt.show()

# a check - each point should appear exactly twice in the edge list
#   this ensures closed loops
assert(set(Counter(itertools.chain.from_iterable(relevant_edges)).values()) == {2})

# now apply a sort of sorts: 
#   break the points into list of lists
#   each inner list is ordered in a iteration of the points

# but first some help
edge_lookup = defaultdict(list)
for p0, p1 in relevant_edges:
    edge_lookup[p0].append(p1)
    edge_lookup[p1].append(p0)

used_points = set()
all_polygons = [] # the outer of the list of lists

for point in relevant_points:
    if point in used_points:
        continue
    
    this_polygon = []
    
    this_polygon.append(point)
    used_points.add(point)
    p = point

    while True:
        m1, m2 = edge_lookup[p]

        if m1 not in used_points:
            this_polygon.append(m1)
            used_points.add(m1)
            p = m1
        elif m2 not in used_points:
            this_polygon.append(m2)
            used_points.add(m2)
            p = m2
        else:
            # both points are used so we are to the end of the chain
            all_polygons.append(this_polygon)
            break

print(f"Resolved: {len(all_polygons)} polygons in the shape")

plt.figure()
plt.scatter(list(map(lambda x: x[0], relevant_points)), list(map(lambda x: x[2], relevant_points)))

for poly in all_polygons:
    x_coords = list(map(lambda x: x[0], poly))
    y_coords = list(map(lambda x: x[2], poly))

    # make the polygon loop on itself for plotting
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    plt.plot(x_coords, y_coords)

plt.axis('equal')

# finally, polygon simplification
old_relevant_points = relevant_points
old_all_polygons = all_polygons
all_polygons = []

def get_unit_vector(p1, p2):
    v = tuple(map(lambda x,y: y-x, p1, p2))
    mag = sum(map(lambda x: x*x, v))**.5
    return tuple(map(lambda x: round(x/mag, 6), v))

for old_polygon in old_all_polygons:
    this_polygon = []

    # has the effect of rotating the polygon start one spot back
    for i in range(len(old_polygon)):
        a = old_polygon[i-2]
        b = old_polygon[i-1]
        c = old_polygon[i]

        if get_unit_vector(a,b) == get_unit_vector(b,c):
            # the point is redundant
            pass
        else:
            # print(f"{get_unit_vector(a,b)} --> {get_unit_vector(b,c)}")
            this_polygon.append(b)

    all_polygons.append(this_polygon)

relevant_points = set(itertools.chain.from_iterable(all_polygons))

old_total_points = sum(map(len, old_all_polygons))
total_points = sum(map(len, all_polygons))

old_total_r_points = len(old_relevant_points)
total_r_points = len(relevant_points)

print(f"Reduced points count from {old_total_points}({old_total_r_points}) to {total_points}({total_r_points})")


plt.figure()
plt.scatter(list(map(lambda x: x[0], relevant_points)), list(map(lambda x: x[2], relevant_points)))

for poly in all_polygons:
    x_coords = list(map(lambda x: x[0], poly))
    y_coords = list(map(lambda x: x[2], poly))

    # make the polygon loop on itself for plotting
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    plt.plot(x_coords, y_coords)

plt.axis('equal')

# now, attempt to create a mask for this item
MASK_RESOLUTION_IN = 0.1
MASK_RESOLUTION_MM = MASK_RESOLUTION_IN * 25.4

min_x = min(map(lambda x: x[0], relevant_points))
max_x = max(map(lambda x: x[0], relevant_points))
min_y = min(map(lambda x: x[2], relevant_points))
max_y = max(map(lambda x: x[2], relevant_points))

min_test_point_x = round(floor(min_x / MASK_RESOLUTION_MM) * MASK_RESOLUTION_MM, 6)
max_test_point_x = round(ceil(max_x / MASK_RESOLUTION_MM) * MASK_RESOLUTION_MM, 6)
min_test_point_y = round(floor(min_y / MASK_RESOLUTION_MM) * MASK_RESOLUTION_MM, 6)
max_test_point_y = round(ceil(max_y / MASK_RESOLUTION_MM) * MASK_RESOLUTION_MM, 6)

print(f"Bonding box: ({min_x}, {min_y}) ({max_x}, {max_y})")
print(f"Test box: ({min_test_point_x}, {min_test_point_y}) ({max_test_point_x}, {max_test_point_y})")

test_points_x = [min_test_point_x]
while test_points_x[-1] < max_test_point_x:
    test_points_x.append(round(test_points_x[-1] + MASK_RESOLUTION_MM, 6))

test_points_y = [min_test_point_y]
while test_points_y[-1] < max_test_point_y:
    test_points_y.append(round(test_points_y[-1] + MASK_RESOLUTION_MM, 6))

test_points_qty_x = round(len(test_points_x))
test_points_qty_y = round(len(test_points_y))

print(f"Total test points grid {test_points_qty_x}x{test_points_qty_y} = {test_points_qty_x * test_points_qty_y}")

# TODO - need to ensure that 0,0 is on the 
# step 1, build the quick collision check map
max_r_sq = max_x + max_y

def isPointInSegDomain(point, seg):
    """
    Return True iff the point is on the segment, endpoint inclusive
    Assumes, but does not check, that point is collinear with the segment endpoints
    Failure to enforce this condition __WILL__ result in incorrect behavior
    """
    p_1, p_2 = seg

    if point == p_1 or point == p_2:
        return True

    # create the vectors from the point to each of the endpoints
    v1 = tuple(map(lambda x,y: x-y, p_1, point))
    v2 = tuple(map(lambda x,y: x-y, p_2, point))

    # check if the x components are opposite directions
    if v1[0] > 0 != v2[0] > 0:
        return True
    
    # edge case if the segment is vertical
    if v1[1] > 0 != v2[1] > 0:
        return True

    return False

def isPointOnSegment(point, seg):
    """
    Return True iff
    1. The point is collinear with the segments
    2. The point is in the domain of the segments, endpoint inclusive
    """

    p_1 = seg[0]
    p_2 = seg[1]
    
    if point == p_1 or point == p_2:
        return True

    v1 = get_unit_vector(point, p_1)
    v2 = get_unit_vector(point, p_2)

    v2_neg = (-v2[0], -v2[1])

    return v1 == v2_neg

def getSegmentIntersection(seg_1, seg_2):
    """Return the point of intersection of the two segments if they interset, otherwise return None"""
    x1 = seg_1[0][0]
    y1 = seg_1[0][1]
    x2 = seg_2[1][0]
    y2 = seg_2[1][1]
    x3 = seg_2[0][0]
    y3 = seg_2[0][1]
    x4 = seg_2[1][0]
    y4 = seg_2[1][1]

    D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)

    a = (x1 * y2 - y1 * x2)
    b = (x3 * y4 - y3 * x4)

    int_x = round((a * (x3 - x4) - b * (x1 - x2)) / D, 6)
    int_y = round((a * (y3 - y4) - b * (y1 - y2)) / D, 6)

    p = (int_x, int_y)

    if isPointInSegDomain(p, seg_1) and isPointInSegDomain(p, seg_2):
        return p
    return None


def isPointInside(point):
    """Return True iff this point is inside the polygon, or on the boundary"""

    test_seg = (point, (-1, -1))

    intersect_point = set()

    for seg in relevant_edges:
        if isPointOnSegment(point, seg):
            return True
        p = getSegmentIntersection(test_seg, seg)
        if p:
            intersect_point.add(p)

    return (len(intersect_point) % 2) == 1

test_regions = []
test_has_intersection = []

for i in range(len(test_points_x)-1):
    for j in range(len(test_points_y)-1):
        a = (test_points_x[i], test_points_y[j])
        b = (test_points_x[i+1], test_points_y[j])
        c = (test_points_x[i+1], test_points_y[j+1])
        d = (test_points_x[i], test_points_y[j+1])

        this_has_intersection = 0
        for p in [a,b,c,d]:
            if isPointInside(p):
                this_has_intersection = 1
                break

        this_region = (a,b,c,d)

        test_regions.append(this_region)
        test_has_intersection.append(this_has_intersection)

        if(len(test_regions) % 128 == 0):
            print(f"Len test_regions = {len(test_regions)}")
plt.figure()

# redraw, just the polygons
for poly in all_polygons:
    x_coords = list(map(lambda x: x[0], poly))
    y_coords = list(map(lambda x: x[2], poly))

    # make the polygon loop on itself for plotting
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    plt.plot(x_coords, y_coords)

x_coords = []
y_coords = []
for test_x, test_y in itertools.product(test_points_x, test_points_y):
    x_coords.append(test_x)
    y_coords.append(test_y)

print("Lag spike incoming")
plt.scatter(x_coords, y_coords, s=[0.25]*len(y_coords))

c_bad = (1.0, 0.0, 0.0, .25)
c_good = (0.0, .5, 0.0, .25)

for this_region, has_intersection in zip(test_regions, test_has_intersection):
    x_coords = list(map(lambda x: x[0], this_region))
    y_coords = list(map(lambda x: x[1], this_region))
    c = c_good if has_intersection == 0 else c_bad

    plt.fill(x_coords, y_coords, facecolor=c)


plt.axis('equal')


plt.show()