
from collections import defaultdict
from numpy import dot
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
plt.show()