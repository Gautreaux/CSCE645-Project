#include "Raster.hpp"

#include <iostream>

Raster::RasterRow::RasterRow(void): data(nullptr), width(0) {}

Raster::RasterRow::RasterRow(const PosType w): 
data(new uint32_t[w]), width(w){}

Raster::RasterRow::RasterRow(const RasterRow& r):
data(new uint32_t[r.width]), width(r.width){
    memcpy(this->data, r.data, sizeof(uint32_t)*width);
}

Raster::RasterRow::RasterRow(RasterRow&& r):
data(r.data), width(r.width) {
    r.data = nullptr;
    r.width = 0;
}

Raster::RasterRow& Raster::RasterRow::operator=(const RasterRow& r){
    if(&r == this){
        return *this;
    }

    delete[] this->data;
    this->width = r.width;
    this->data = new uint32_t[width];
    memcpy(this->data, r.data, sizeof(uint32_t)*width);
    return *this;
}

Raster::RasterRow& Raster::RasterRow::operator=(RasterRow&& r){
    if(&r == this){
        return *this;
    }

    delete[] this->data;
    this->width = r.width;
    this->data = r.data;
    r.data = nullptr;
    r.width = 0;
    return *this;
}

Raster::RasterRow::~RasterRow(void){
    delete[] this->data;
}

Raster::Raster(void): width(0), height(0), data(nullptr){};

Raster::Raster(const PosType w, const PosType h): 
width(w), height(h), data(new RasterRow[h])
{
    for(unsigned int i = 0; i < h; i++){
        data[i] = std::move(RasterRow(w));
    }
};

Raster::Raster(const Raster& r): 
width(r.width), height(r.height), data(new RasterRow[r.height]){
    for(unsigned int i = 0; i < r.height; i++){
        data[i] = r[i];
    }
}

Raster::Raster(Raster&& r): 
width(r.width), height(r.height), data(r.data){
    r.data = nullptr;
    r.width = 0;
    r.height = 0;
}

Raster::~Raster(void){
    delete[] this->data;
}