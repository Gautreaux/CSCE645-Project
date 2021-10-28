
from genericpath import isdir
import os

# location to dump files to
DUMP_DIRECTORY = "dump/"
DUMP_LOCATION_SLDPRT = f"{DUMP_DIRECTORY}sldprt/"
DUMP_LOCATION_STL = f"{DUMP_DIRECTORY}stl/"

# precision to use in all round statements
ROUND_PRECISION = 10

# number of test points per inch
RASTER_RESOLUTION_LOW = 10
RASTER_RESOLUTION_MEDIUM = 50
RASTER_RESOLUTION_HIGH = 200
RASTER_RESOLUTION_ULTRA = 500

_cwd = os.getcwd()
_dir_list = [DUMP_DIRECTORY, DUMP_LOCATION_SLDPRT, DUMP_LOCATION_STL]

# creates the expected directories that may not be present
for d in _dir_list:
    p = f"{_cwd}/{d}"
    if not os.path.isdir(p):
        os.mkdir(p)
