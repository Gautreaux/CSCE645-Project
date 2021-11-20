#include "Raster.cuh"

#include <iostream>

Raster::Raster(void): width(0), height(0), data(nullptr){};

Raster::Raster(const PosType w, const PosType h): width(w), height(h)
{
    this->data = new uint32_t[this->getLinearDataSize()];
    memset(this->data, 0, this->getLinearDataSize() * sizeof(uint32_t));
};

Raster::Raster(const Raster& r): width(r.width), height(r.height){
    this->data = new uint32_t[this->getLinearDataSize()];
    memcpy(this->data, r.data, this->getLinearDataSize() * sizeof(uint32_t));
}

Raster::Raster(Raster&& r): width(r.width), height(r.height){
    this->data = r.data;
    r.data = nullptr;
    r.width = 0;
    r.height = 0;
}

Raster::~Raster(void){
    delete[] this->data;
}