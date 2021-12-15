# simple script for test/simulating/verifying memory access patterns

from math import ceil
import numpy

SAMPLES_PER_IN = 170

SHEET_WIDTH_IN = 48
SHEET_HEIGHT_IN = 24

PART_WIDTH_IN = 6
PART_HEIGHT_IN = 12

SHEET_WIDTH = SHEET_WIDTH_IN * SAMPLES_PER_IN
SHEET_HEIGHT = int(ceil(SHEET_HEIGHT_IN * SAMPLES_PER_IN / 32))

PART_WIDTH = PART_WIDTH_IN * SAMPLES_PER_IN
PART_HEIGHT = int(ceil(PART_HEIGHT_IN * SAMPLES_PER_IN / 32))

# FOR SPECIFIC TEST
SHEET_HEIGHT = 5
SHEET_WIDTH = 7
PART_HEIGHT = 3
PART_WIDTH = 3

# logging
print(f"Sheet: {SHEET_WIDTH} ({SHEET_WIDTH_IN}) X {SHEET_HEIGHT} i32 ({SHEET_HEIGHT_IN})")
print(f"Part: {PART_WIDTH} ({PART_WIDTH_IN}) X {PART_HEIGHT} i32 ({PART_HEIGHT_IN})")

# first, determine the span of the parts
output_height = SHEET_HEIGHT - PART_HEIGHT + 1
output_width = SHEET_WIDTH - PART_WIDTH + 1

print(f"Placement x:[0, {output_height-1}]; Pacement y:[0, {output_width-1}]")

# prepare the output array size
output_array = [None]*output_height
for i in range(output_height):
    output_array[i] = [""]*output_width

# and an alternative output format
#   in a single iterable
output_linear = []

def printOutputArray() -> None:
    """Print output array, but in y-pos format"""
    if output_height > 15 or output_width > 15:
        print(f"output_array<{output_height}hX{output_width}w")
        return

    for i in range(output_height):
        print(output_array[-(i+1)])

# now determine how the thing is built
for o_i in range(output_height):
    for o_j in range(output_width):
        points = []

        for p_i in range(PART_HEIGHT):
            for p_j in range(PART_WIDTH):
                s_i = o_i + p_i
                s_j = o_j + p_j

                assert(s_i < SHEET_HEIGHT)
                assert(s_j < SHEET_WIDTH)

                points.append(f"P{p_i}:{p_j}S{s_i}:{s_j}")
        
        output_array[o_i][o_j] = "+".join(points)
        output_linear.append((f"O{o_i}:{o_j}", output_array[o_i][o_j]))

# printOutputArray()

access_s21 = []

for o,k in output_linear:
    if "S2:1" in k:
        access_s21.append(o)

print(access_s21)
