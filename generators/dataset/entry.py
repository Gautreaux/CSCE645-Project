import time
from typing import Any, Tuple

from generators.dataset.env import RASTER_RESOLUTION_HIGH
from generators.dataset.my_types import EDGE_SELECTOR_UNIQUE, TRANSFORM_KEEP_XZ
from generators.dataset.polygon_utils import buildPolygonFromEdges
from generators.dataset.raster_utils import rasterAlignPolygon, constructRasterFromPolygon
from generators.dataset.stl_utils import buildPointSetFromEdges, determineIfLaserNormal, doEdgeReduction, loadSTLData, projectPointsEdgesOntoPlane, transformPoints


def preprocessFile(filePath: str, resolution: int = RASTER_RESOLUTION_HIGH) -> Tuple[Any, Any]:
    """
    Preprocess the file specified by filepath 
    return a tuple of Polygon and Rasterized result
    """

    print(f"Preprocess: {filePath}...\t\t", end="")

    start_time = time.time()

    s = loadSTLData(filePath)

    n = determineIfLaserNormal(s)

    if n == None:
        print(" Non-Laserable Exception")
        raise NotImplementedError("Non-laserable files noy yet handled")
    if n != (0, 1.0, 0):
        print(" Non-y Normal")
        raise NotImplementedError("Non-y normal not implemented")

    ps, ed = projectPointsEdgesOntoPlane(s, n, keep_planar=True, keep_non_planar=False)

    for k in ed:
        assert(len(k) == 2)
        assert(len(k[0]) == 3)
        assert(len(k[1]) == 3)

    ps, ed = transformPoints(ps, TRANSFORM_KEEP_XZ, ed)

    ed = doEdgeReduction(ed, EDGE_SELECTOR_UNIQUE)
    ps = buildPointSetFromEdges(ed)

    sh_poly = buildPolygonFromEdges(ed)

    raster_poly = rasterAlignPolygon(sh_poly, resolution)
    raster_out = constructRasterFromPolygon(raster_poly, resolution, pre_scaled=True)

    end_time = time.time()

    print(f"Total time: {round(end_time - start_time, 3)}s")

    return (raster_poly, raster_out)