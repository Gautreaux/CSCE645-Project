
from genericpath import isdir
import os

# location to dump files to
DUMP_DIRECTORY = "dump/"
DUMP_LOCATION_SLDPRT = f"{DUMP_DIRECTORY}sldprt/"
DUMP_LOCATION_STL = f"{DUMP_DIRECTORY}stl/"

_cwd = os.getcwd()
_dir_list = [DUMP_DIRECTORY, DUMP_LOCATION_SLDPRT, DUMP_LOCATION_STL]

for d in _dir_list:
    p = f"{_cwd}/{d}"
    if not os.path.isdir(p):
        os.mkdir(p)
