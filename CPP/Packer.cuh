#pragma once
#include "pch.h"

#include "Raster.cuh"


// bake the part raster into the space raster at postition p
//  available both in the device and global space
__global__ void bakeRaster(const Raster* const part, Raster* const space, const Position p);

// find the first position in the space raster that part raster fits
//  available in both the device and global space
//  returns the position that was found or returns <PosType::max, PosType::max>
__global__ void checkRaster(const Raster* const part, const Raster* const space, Position* const p_out);