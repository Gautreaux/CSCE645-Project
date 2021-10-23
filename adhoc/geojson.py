# needs to run in WSL because idk

from rasterio.features import rasterize
import json
from numpy import flipud


with open("geojson.json") as in_file:
    geo = json.load(in_file)

# print(geo)

new_list = []
old_list = geo["coordinates"][0]
geo["coordinates"][0] = new_list

# TODO - these could be much more efficient
min_x = min(map(lambda x: x[0], old_list))
min_y = min(map(lambda x: x[1], old_list))

for x,y in old_list:
    new_list.append([(x - min_x)*10, (y - min_y)*10])

print(new_list)

f = rasterize([geo], out_shape=(26,31), all_touched=True)
f = flipud(f)
print(f)
