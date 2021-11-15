#include "raster.hpp"

#include <string.h>

Raster::Raster(void) : width_exact(0), width(0), height(0), resolution(0), mem_buffer(nullptr){}

Raster::Raster(const Raster& r) : 
width_exact(r.width_exact), 
width(r.width),
height(r.height),
resolution(r.resolution),
mem_buffer(nullptr)
{
    const auto c = r.width * r.height;
    mem_buffer = new uint32_t[c];
    memcpy(mem_buffer, r.mem_buffer, c*sizeof(uint32_t));
}

Raster& Raster::operator=(const Raster& r){
    if(&r == this){
        return *this;
    }

    delete this->mem_buffer;
    memcpy(this, &r, sizeof(Raster));

    const auto c = r.width * r.height;
    mem_buffer = new uint32_t[c];
    memcpy(mem_buffer, r.mem_buffer, c*sizeof(uint32_t));

    return *this;
}

Raster::Raster(Raster&& r){
    memcpy(this, &r, sizeof(Raster));
    memset(&r, 0, sizeof(Raster));
}

Raster& Raster::operator=(Raster&& r){
    if(this == &r){
        return *this;
    }

    delete mem_buffer;

    memcpy(this, &r, sizeof(Raster));
    memset(&r, 0, sizeof(Raster));

    return *this;
}