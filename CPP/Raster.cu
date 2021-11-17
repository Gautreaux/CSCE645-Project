#include "Raster.cuh"

Raster::Raster(const PosType w, const PosType h): width(w), height(h)
{
    const auto num_bytes = this->getLinearDataSize() * sizeof(uint32_t);

    this->data = (uint32_t*)malloc(num_bytes);

    memset(this->data, 0, num_bytes);
};

Raster::~Raster(void){
    free(this->data);
}