#include <cstdint>

class Raster{
private:
    uint16_t width_exact; //width in number of bits
    uint16_t width; // ceil(width / 32), size in memory
    uint16_t height; // aka number rows
    uint16_t resolution; // the resolution if provided, 0 if not set
    uint32_t* mem_buffer; // memory pointer to where the raster is actually living
public:
    Raster(void);
    Raster(const Raster& r);
    Raster(Raster&& r);

    Raster& operator=(const Raster& r);
    Raster& operator=(Raster&& r);

};