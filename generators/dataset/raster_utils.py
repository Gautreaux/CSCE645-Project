
from math import floor, ceil
from numpy import flipud, uint8
import numpy
from rasterio.features import rasterize
from shapely import affinity
from shapely.geometry import mapping as ShapelyToGeoJSON
import struct
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


def compressRasterFormat(raster) -> Any:
    """Compresses the raster fromat by using int8 cells in the rows"""
    
    new_raster = []

    for row in raster:
        this_row = []

        for i, v in enumerate(row):
            m = i % 8
            if m == 0:
                this_row.append(0)
            this_row[-1] = this_row[-1] | (v << m)

        new_raster.append(this_row)
    return numpy.array(new_raster)


def expandRasterFormat(raster) -> Any:
    """Expand the raster format from bits to 1,0 array"""

    new_raster = numpy.zeros((len(raster), len(raster[0])*8), dtype=uint8)

    for i, row in enumerate(raster):
        for j,v in enumerate(row):
            for o in range(8):
                new_raster[i][j*8+o] = (1 if (v & (1<<o)) else 0)
        
    while sum(new_raster[:, -1]) == 0:
        new_raster = new_raster[:, :-1]
    
    return new_raster


def exportRasterToFile(raster, filePath: str, pre_compressed: bool=False) -> None:
    """Exports a raster to a given filepath"""

    if not pre_compressed:
        raster = compressRasterFormat(raster)

    num_rows, num_bytes_per_row = raster.shape

    with open(filePath, 'wb') as outfile:
        outfile.write(struct.pack(">I", num_rows))
        outfile.write(struct.pack(">I", num_bytes_per_row))

        for cell in numpy.nditer(raster):
            outfile.write(struct.pack("B", cell))


def importRasterFromFile(filePath: str, decompress: bool = False) -> Any:
    
    with open(filePath, 'rb') as infile:
        buffer = infile.read()

    (num_rows, ) = struct.unpack_from(">I", buffer)
    (num_bytes_per_row, ) = struct.unpack_from(">I", buffer, offset=4)

    a = numpy.zeros(num_rows * num_bytes_per_row, dtype=uint8)

    for i in range(num_rows * num_bytes_per_row):
        a[i] = (struct.unpack_from("B", buffer, 8 + i))[0]

    a = a.reshape((num_rows, num_bytes_per_row))


    if decompress:
        return expandRasterFormat(a)
    else:
        return a
