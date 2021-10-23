
from math import floor, ceil
from numpy import flipud
from rasterio.features import rasterize
from shapely import affinity
from shapely.geometry import mapping as ShapelyToGeoJSON
from typing import Any, Tuple

from generators.dataset.env import ROUND_PRECISION
from generators.dataset.my_types import ShapelyPolygon



def calculateRasterBounds(
    polygon: ShapelyPolygon,
    test_points_per_inch: int,
) -> Tuple[float, float, float, float, int, int]:
    """
    Calculates the raster bounds of the polygon

    returns a 6-tuple ordered:
        raster_x_min, raster_y_min, raster_x_max, raster_y_max, a, b

        where a,b are the number of raster points in the region:
            a = (raster_x_max - raster_x_min) * test_points_per_inch
    """

    min_x, min_y, max_x, max_y = polygon.bounds

    e = floor(min_x * test_points_per_inch)
    f = floor(min_y * test_points_per_inch)
    g = ceil(max_x * test_points_per_inch)
    h = ceil(max_y * test_points_per_inch)

    raster_min_x = round(e / test_points_per_inch, ROUND_PRECISION)
    raster_min_y = round(f / test_points_per_inch, ROUND_PRECISION)
    raster_max_x = round(g / test_points_per_inch, ROUND_PRECISION)
    raster_max_y = round(h / test_points_per_inch, ROUND_PRECISION)

    a = g - e
    b = h - f

    # assert(a == round((raster_max_x - raster_min_x) * test_points_per_inch))
    # assert(b == round((raster_max_y - raster_min_y) * test_points_per_inch))

    return (raster_min_x, raster_min_y, raster_max_x, raster_max_y, a, b)


def rasterAlignPolygon(
    polygon: ShapelyPolygon,
    test_points_per_inch: int,
) -> ShapelyPolygon:
    """
    Scale the polygon to prepare for rasterization
    
    Required behaviors
    1. Translate the polygon such that it lies entirely
        inside the first quadrant (positive x,y)
    2. Scale the polygon such that one unit in the original polygon
        corresponds to test_points_per_inch units in the returned polygon
    3. Translate the polygon such that it is aligned to a multiple of 
        test_points_per_inch and tight against the x,y axis.
        Notice that this means the polygon will probably
        not be incident with y = 0 nor x = 0
    #1 and #3 occur concurrently with a single affine transform, then #2
    """

    (a, b, _, _, _, _) = calculateRasterBounds(polygon, test_points_per_inch)

    translate_x = -a
    translate_y = -b
    scale_factor = test_points_per_inch

    p = affinity.affine_transform(polygon, [1, 0, 0, 1, translate_x, translate_y])
    return affinity.scale(p, xfact=scale_factor, yfact=scale_factor, origin=(0, 0, 0))


def constructRasterFromPolygon(
    polygon: ShapelyPolygon, 
    test_points_per_inch: int,
    pre_scaled: bool = False,
) -> Any:
    """
    Construct a 2d raster of this polygon with number of points per inch
    The raser is aligned to multiples of test_point_per_inch

    The polygon needs to be scaled such that 1 unit of the polygon
        corresponds to 1 / test_points_per_inch in the actual polygon
    The polygon also need to be entirely in first quadrant (positive x,y)
    The function will perform this scaling/translation. This behavior can be
        turned off by setting `pre_scaled` to True

    Returns:
        a two-dimensional array of 0,1 values indicating if 
            0 : no part of the polygon enters this region
            1 : any part of the polygon enters this region

        the array is indexed in row major order
    """

    if not pre_scaled:
        polygon = rasterAlignPolygon(polygon, test_points_per_inch)

    (_, _, max_x, max_y) = polygon.bounds

    b = ceil(max_y)
    a = ceil(max_x)

    f = rasterize([ShapelyToGeoJSON(polygon)], fill=0, out_shape=(b,a), all_touched=True)

    return f
