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