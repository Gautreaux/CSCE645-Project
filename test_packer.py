import os
import cv2
import numpy
from PIL import Image
from generators.dataset.env import RASTER_RESOLUTION_HIGH

from generators.dataset.raster_utils import importRasterFromPlaintext



RASTER_DIR = "adhoc/MarbleSorter/Raster"
f_names = [f for f in os.listdir(RASTER_DIR) if f.rpartition(".")[-1] == "raster"]

raster_list = []

for f in f_names:
    print(f"Importing: {f}")
    r = importRasterFromPlaintext(f"{RASTER_DIR}/{f}")
    raster_list.append(r)
    
    if len(raster_list) > 5:
        break

sheet = numpy.ones((RASTER_RESOLUTION_HIGH * 24, RASTER_RESOLUTION_HIGH*48), dtype=numpy.uint8)

for r in raster_list:
    r = r.astype(numpy.uint8)
    r = r * 255
    res = cv2.matchTemplate(sheet, r, cv2.TM_SQDIFF, None)


    threshold = 1
    loc = numpy.where(res >= threshold)

    # this is not working
    if loc:
        # there is a point to put this one in
        print(len(loc))
        p = (loc[0][0], loc[1][0])
        print(p)

    sheet[p[1]:p[1]+r.shape[0], p[0]:p[0]+r.shape[1]] = r

s_small = cv2.resize(sheet, (1600, 800))
cv2.imshow("ALL", s_small)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image