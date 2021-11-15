#include "Packer.cuh"


__global__ void bakeRaster(const Raster* const part, Raster* const space, const Position p){
    return;
}

__global__ void checkRaster(const Raster* const part, const Raster* const space, Position* const p_out){
    // *p_out = Position(-1U, -1U);
    p_out->first = -1U;
    p_out->second = -1U;
}