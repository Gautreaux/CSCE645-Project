

from generators.dataset.render_utils import plotPointEdge2D
from generators.dataset.stl_utils import determineIfLaserNormal, loadSTLData, projectPointsEdgesOntoPlane, transformPoints


FILE_PATH = "adhoc/basebot.stl"

s = loadSTLData(FILE_PATH)

n = determineIfLaserNormal(s)

assert(n != None)

print(f"Found laser normal {n}")

ps, ed = projectPointsEdgesOntoPlane(s, n, keep_planar=True, keep_non_planar=False)

for k in ed:
    assert(len(k) == 2)
    assert(len(k[0]) == 3)
    assert(len(k[1]) == 3)

ps, ed = transformPoints(ps, (lambda x: (x[0], x[2])), ed)

plotPointEdge2D(ps, ed, 'Pre-Reduce', True)

