#pragma once
#include "pch.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <string.h> // for memset


using PosType = uint16_t;
using Position = std::pair<PosType, PosType>;

class Raster{
public:
    class RasterRow{
    protected:
        uint32_t* data;
        PosType width; // ram is infinite right?

    public:
        RasterRow(void);
        RasterRow(const PosType w);
        RasterRow(const RasterRow& r);
        RasterRow(RasterRow&& r);
        RasterRow& operator=(const RasterRow& r);
        RasterRow& operator=(RasterRow&& r);
        ~RasterRow(void);

        inline uint32_t& operator[](const int index){
            return data[index];
        }
        inline const uint32_t& operator[](const int index) const {
            return data[index];
        }
    };
protected:
    PosType width;
    PosType height; // exact height, round up to next multiple 32
    RasterRow* data;
public:
    Raster(void);
    Raster(const PosType w, const PosType h);
    Raster(const Raster& r);
    Raster(Raster&& r);
    ~Raster(void);

    // return the size of the raster: x*y
    inline size_t getSize(void) const {return width * height;}

    // return the size of the raster packed into linear memory of uint32_t chunks
    //  according
    inline size_t getLinearDataSize(void) const {
        const auto height_32 = (height >> 5) + (height & (0b11111) ? (1) : (0));
        return height_32 * width; 
    }

    inline PosType getWidth(void) const {return width;}
    inline PosType getHeight(void) const {return height;}

    inline RasterRow& operator[](const int index){
        return data[index >> 5]; // div by 32 to get proper height
    }

    inline const RasterRow& operator[](const int index) const {
        return data[index >> 5]; // div by 32 to get proper height
    }
};