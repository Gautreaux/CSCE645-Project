# testing some things

from svgpathtools import (
    svg2paths, 
    path_encloses_pt, 
    disvg
)

from math import floor, ceil

# draw-bot.svg - the source file
# dn_sample_1.svg was a deep nest on the sample file
#   gravity nest - prioritizing area
#   12.6 high (fixed) solved to ~8.04 width 
# This is a total area of ~101.304 in^2
# Question: Can we do better?

# note, svg took about 1 min to produce
#   this will probably take much longer

SVG_SOURCE = "draw-bot.svg"
SVG_SOLUTION = "dn_sample_1.svg"

SOURCE_NUM_PARTS = 20

MATERIAL_HEIGHT_IN = 12.6
PTS_PER_IN = 90

RASTER_RESOLUTION_IN = 0.005
RASTER_RESOLUTION_PTS = RASTER_RESOLUTION_IN * PTS_PER_IN

path, _ = svg2paths(SVG_SOURCE)

subset_types = set()

for p in path:
    for k in p:
        subset_types.add(type(k))
    
    # this will be a function at some point
    xmin, xmax, ymin, ymax = p.bbox()

    x_pt_min = floor(xmin / RASTER_RESOLUTION_PTS) * RASTER_RESOLUTION_PTS
    y_pt_min = floor(ymin / RASTER_RESOLUTION_PTS) * RASTER_RESOLUTION_PTS
    x_pt_max = ceil(xmax / RASTER_RESOLUTION_PTS) * RASTER_RESOLUTION_PTS
    y_pt_max = ceil(ymax / RASTER_RESOLUTION_PTS) * RASTER_RESOLUTION_PTS

    # a visualizer would be nice
    # doesnt work - needs to be a continuous path
    assert(path_encloses_pt(complex(x_pt_min, y_pt_min), 0+0j, p) == False)
    assert(path_encloses_pt(complex(x_pt_max, y_pt_max), 0+0j, p) == False)
    print((x_pt_min, x_pt_max, y_pt_min, y_pt_max))



print(len(path))
print(subset_types)
