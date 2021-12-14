#include "Raster.hpp"

#include <iostream>

Raster::RasterRow::RasterRow(void): data(nullptr), width(0) {}

Raster::RasterRow::RasterRow(const PosType w): 
data(new uint32_t[w]), width(w){
    memset(this->data, 0, sizeof(uint32_t)*width);
}

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
width(w), height(h), data(new RasterRow[CEIL_DIV_32(h)])
{
    for(unsigned int i = 0; i < CEIL_DIV_32(h); i++){
        data[i] = std::move(RasterRow(w));
    }
};

Raster::Raster(const Raster& r): 
width(r.width), height(r.height), data(new RasterRow[CEIL_DIV_32(r.height)]){
    for(unsigned int i = 0; i < CEIL_DIV_32(r.height); i++){
        data[i] = r.data[i];
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

char* Raster::linearPackData(void) const {
    char* c = (char*)malloc(this->getLinearDataSize()*sizeof(uint32_t));

    const size_t row_len_bytes = width * sizeof(uint32_t);

    for(unsigned int i = 0; i < CEIL_DIV_32(height); i++){
        memcpy(c+(i*getLinearPackStride()), data[i].getData(), row_len_bytes);
    }

    return c;
}

unsigned char Raster::getBit(const int x, const int y) const {
    assert(x >= 0);
    assert(x < width);
    assert(y >= 0);
    assert(y < height);
    uint32_t cell = data[FAST_DIV_32(y)][x];
    return (cell & (1 << FAST_MOD_32(y))) >> (FAST_MOD_32(y));
}

void Raster::setBit(const int x, const int y){
    assert(x >= 0);
    assert(x < width);
    assert(y >= 0);
    assert(y < height);
    data[FAST_DIV_32(y)][x] |= (1 << FAST_MOD_32(y));
}

void Raster::clearBit(const int x, const int y){
    assert(x >= 0);
    assert(x < width);
    assert(y >= 0);
    assert(y < height);
    data[FAST_DIV_32(y)][x] &= (~(1 << FAST_MOD_32(y)));
}

void Raster::print(
    const PosType max_x, const PosType max_y,
    const PosType min_x, const PosType min_y
) const {
    if (max_x == (PosType)~0){
        const_cast<PosType&>(max_x) = this->width;
    }
    if (max_y == (PosType)~0){
        const_cast<PosType&>(max_y) = this->height;
    }

    printf("======Raster %u x %u (%u) [%u %u] ========\n", max_x - min_x, CEIL_DIV_32(max_y) - CLAMP_32(min_y), max_y - min_y, this->width, this->height);

    for(unsigned int i = CLAMP_32(min_y) ; i < CEIL_DIV_32(max_y); i++){
        // print a single row
        const auto data_row = data[i].getData();
        for(unsigned int j = min_x; j < max_x; j++){
            printf("%08X ", data_row[j]);
        }
        printf("\n");
    }


    printf("================================\n");
}