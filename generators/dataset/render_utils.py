"""Various utility functions for rendering out the data"""

from matplotlib import pyplot as plt
from generators.dataset.my_types import EdgeDict, PointSet


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