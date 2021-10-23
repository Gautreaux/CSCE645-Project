# needs to run in WSL because idk

from rasterio.features import rasterize
import json
from numpy import ceil, flipud, floor


with open("geojson.json") as in_file:
    geo = json.load(in_file)

# print(geo)

new_list = []
old_list = geo["coordinates"][0]
geo["coordinates"][0] = new_list

# TODO - these could be much more efficient
min_x = min(map(lambda x: x[0], old_list))
min_y = min(map(lambda x: x[1], old_list))
max_x = max(map(lambda x: x[0], old_list))
max_y = max(map(lambda x: x[1], old_list))

TEST_RESOLUTION = .1
min_x = floor(min_x / TEST_RESOLUTION) * TEST_RESOLUTION
min_y = floor(min_y / TEST_RESOLUTION) * TEST_RESOLUTION

for x,y in old_list:
    new_list.append([(x - min_x)*10, (y - min_y)*10])

print(new_list)

f = rasterize([geo], fill=0, out_shape=(26,31), all_touched=True)
f = flipud(f)
print(f)


print((min_x, min_y, max_x, max_y))
min_x = min(map(lambda x: x[0], new_list))
min_y = min(map(lambda x: x[1], new_list))
max_x = max(map(lambda x: x[0], new_list))
max_y = max(map(lambda x: x[1], new_list))

print((min_x, min_y, max_x, max_y))