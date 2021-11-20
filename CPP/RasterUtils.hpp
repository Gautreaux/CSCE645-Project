#pragma once

#include "pch.h"
#include "Raster.hpp"

#include <fstream>
#include <string>
#include <string.h> // for memcpy

// read a raster file from the provided filename
Raster readRaster(const std::string& file_path, const bool is_compressed = false);

// pack a raster into the buffer pointer to 
//  return number of uint32_t packed 
//      another raster could be packed at (buffer + int)
int packRaster(const Raster& r, int32_t * const buffer);

// bake part into dest at x,y position
//  basically overlay part onto dest with bitwise or
// NOTE - currently does not support bitwise checking
void bakeRaster_cpu(
    const Raster& part, Raster& space, 
    const PosType x, const PosType y,
    const uint8_t y_offset
);

// determine all the valid positions in space that part can be placed
//  all the positions where part will not collide with space
//      very, very slow operation, prime for GPU acceleration
// returns a 1-bit where collisions occur
// NOTE - currently does not support bitwise checking
Raster buildPackMap_cpu(const Raster& space, const Raster& part);

// return the best position and offset in the space map
//  best is the leftmost and ties broken by down-most
//   minimize out_x then out_y
// NOTE - currently does not support bitwise checking
void pickBestPosition_cpu(
    const Raster& space,
    PosType& out_x, PosType& out_y,
    uint8_t& out_y_offset
);