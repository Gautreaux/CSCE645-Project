"""Various utility functions for rendering out the data"""

from matplotlib import pyplot as plt
from typing import Optional

from generators.dataset.my_types import EdgeDict, PointSet, ShapelyPolygon

from shapely import affinity


def plotPointEdge2D(
    points: PointSet, 
    edges: EdgeDict, 
    figure_name=None, 
    show_result: bool = False
) -> None:
    """
    Use MatPlotLib to plot the points and edges onto a figure
    Can specify a figure name to plot onto a specific figure
    """

    # enforce that these are 2d points
    assert(len(next(iter(points))) == 2)

    plt.figure(figure_name)

    x_pts = []
    y_pts = []

    for x,y in points:
        x_pts.append(x)
        y_pts.append(y)

    plt.scatter(x_pts, y_pts)

    plt.axis("equal")

    for ((a,c), (b,d)) in edges:
        plt.plot((a,b), (c,d))

    if show_result:
        plt.show()


def plotPolygon(
    sh_poly: ShapelyPolygon,
    figure_name=None,
    show_result: bool = False,
) -> None:
    """Plot a ShapelyPolygon to figure with figure_name"""
    plt.figure(figure_name)

    x,y = sh_poly.exterior.xy
    plt.plot(x,y, c="blue")

    for i in sh_poly.interiors:
        x,y = i.xy
        plt.plot(x,y, c="red")

    plt.axis("equal")

    if show_result:
        plt.show()


def drawPolygonOnRaster(
    raster,
    sh_poly: Optional[ShapelyPolygon] = None,
    figure_name=None,
    show_result: bool = False,
) -> None:
    """
    Draw a raster to figure with figure_name
    Optionally, if a shapely_polygon is provided,
        draw that over the raster on the same figure
        Must be pre-scaled to the match the raster

    NOTE: if a polygon is provided, it should match 
        the one used to construct the raster
        HOWEVER: at render time, another translation 
        of [-.5, -.5] is applied so that the polygon
        will match the rendering output of the raster

        DO NOT USE THIS OUTPUT FOR EXACTNESS
    """

    plt.figure(figure_name)
    plt.imshow(raster, interpolation="nearest")

    if sh_poly:
        p = affinity.affine_transform(sh_poly, [1, 0, 0, 1, -.5, -.5])
        plotPolygon(p, figure_name)

    plt.axis("equal")

    if show_result:
        plt.show()
