import time

from py.dataset.env import RASTER_RESOLUTION_LOW, RASTER_RESOLUTION_ULTRA
from py.dataset.my_types import EDGE_SELECTOR_UNIQUE, TRANSFORM_KEEP_XZ
from py.dataset.polygon_utils import buildPolygonFromEdges
from py.dataset.raster_utils import calculateRasterBounds, compressRasterFormat, constructRasterFromPolygon, expandRasterFormat, exportRasterToFile, importRasterFromFile, rasterAlignPolygon
from py.dataset.render_utils import drawPolygonOnRaster, plotPointEdge2D, plotPolygon
from py.dataset.stl_utils import buildPointSetFromEdges, determineIfLaserNormal, doEdgeReduction, loadSTLData, projectPointsEdgesOntoPlane, transformPoints


FILE_PATH = "adhoc/basebot.stl"

start_time = time.time()

s = loadSTLData(FILE_PATH)

n = determineIfLaserNormal(s)

assert(n != None)

print(f"Found laser normal {n}")

ps, ed = projectPointsEdgesOntoPlane(s, n, keep_planar=True, keep_non_planar=False)

for k in ed:
    assert(len(k) == 2)
    assert(len(k[0]) == 3)
    assert(len(k[1]) == 3)

ps, ed = transformPoints(ps, TRANSFORM_KEEP_XZ, ed)

plotPointEdge2D(ps, ed, 'Pre-Reduce', False)

ed = doEdgeReduction(ed, EDGE_SELECTOR_UNIQUE)
ps = buildPointSetFromEdges(ed)

plotPointEdge2D(ps, ed, 'Post-Reduce', False)

sh_poly = buildPolygonFromEdges(ed)

plotPolygon(sh_poly, "Shapely Polygon", False)

raster_poly = rasterAlignPolygon(sh_poly, RASTER_RESOLUTION_LOW)
raster_out = constructRasterFromPolygon(raster_poly, RASTER_RESOLUTION_LOW, pre_scaled=True)

end_time = time.time()

# print(sh_poly.bounds)
# print(calculateRasterBounds(sh_poly, RASTER_RESOLUTION_LOW))
# print(raster_poly.bounds)
# print(raster_out.shape)

r = compressRasterFormat(raster_out)
r_x = expandRasterFormat(r)

exportRasterToFile(raster_out, "tmp.rasterout")
r_in = importRasterFromFile("tmp.rasterout")

assert((raster_out == r_x).all())
assert((r_in == r).all())
assert((expandRasterFormat(r_in) == raster_out).all())

drawPolygonOnRaster(raster_out, raster_poly, "Poly+Raster", True)

print(f"Time to raster including disk read and some drawing: {round(end_time - start_time, 4)}s")


start_time = time.time()

raster_poly = rasterAlignPolygon(sh_poly, RASTER_RESOLUTION_ULTRA)
raster_out = constructRasterFromPolygon(raster_poly, RASTER_RESOLUTION_ULTRA, pre_scaled=True)

end_time = time.time()

print(f"Time to construct an ultra resolution raster: {round(end_time - start_time, 4)}s")
