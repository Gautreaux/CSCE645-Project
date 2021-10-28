import os
import shutil
import time
import svgpathtools

from generators.dataset.entry import preprocessFile
from generators.dataset.env import RASTER_RESOLUTION_HIGH
from generators.dataset.raster_utils import exportRasterToPlaintext

FILE_DIR = "adhoc/MarbleSorter/STL"
OUT_DIR = "adhoc/MarbleSorter/Raster"

start_time = time.time()
success = 0
total = 0


for f in os.listdir(FILE_DIR):
    n, _, t = f.rpartition(".")
    if t != "stl":
        continue

    total += 1

    try:
        p, r = preprocessFile(f"{FILE_DIR}/{f}", resolution=RASTER_RESOLUTION_HIGH)
    except NotImplementedError:
        continue

    success += 1

    fp = f"{OUT_DIR}/{n}.raster"

    with open(fp, 'w') as out_file:
        exportRasterToPlaintext(r, fp, resolution=RASTER_RESOLUTION_HIGH)

    svg_path = f"{OUT_DIR}/{n}.svg"

    # print(p.svg())

    svg_str = p.svg()

    while(svg_str[0] != 'd'):
        svg_str = svg_str.partition(" ")[-1]
    l, _, _ = svg_str.rpartition("z")
    svg_str = f"{l[3:]}z"

    # print(svg_str)
    # print(t in svg_str)

    svg_obj = svgpathtools.parse_path(svg_str)

    svgpathtools.wsvg([svg_obj], filename=svg_path)

    shutil.copy(f"{FILE_DIR}/{f}", f"{OUT_DIR}/{f}")

end_time = time.time()

print(f"Total time for {success}/{total} files: {round(end_time - start_time, 3)}")