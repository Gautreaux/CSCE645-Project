#pragma once
#include "pch.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <string.h> // for memset


using PosType = uint16_t;
using Position = std::pair<PosType, PosType>;

// basically a 2D bit map
class Raster{
public:
    class RasterRow{
    protected:
        uint32_t* data;
        PosType width; // ram is infinite right?

    public:
        explicit RasterRow(void);
        explicit RasterRow(const PosType w);
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

        inline uint32_t* getData(void) const {
            return data;
        }
    };
protected:
    PosType width;
    PosType height; // exact height, round up to next multiple 32
    RasterRow* data;
public:
    explicit Raster(void);

    // preallocate with given width and height
    //  and all zeros therein
    Raster(const PosType w, const PosType h);

    Raster(const Raster& r);
    Raster(Raster&& r);
    ~Raster(void);

    // return the size of the raster: x*y
    inline size_t getSize(void) const {return width * height;}
    
    // return the size of the raster packed into linear memory of uint32_t chunks
    //  the number of uint32_t registers needed for this raster
    inline size_t getLinearDataSize(void) const {
        return CEIL_DIV_32(height) * width; 
    }

    // return the length of a single row packed into linear memory of uint32_t chunks
    //  including any tail padding
    //  returns in bytes
    inline size_t getLinearPackStride(void) const{
        return width * sizeof(uint32_t);
    }

    inline PosType getWidth(void) const {return width;}
    inline PosType getHeight(void) const {return height;}

    unsigned char getBit(const int x, const int y) const;

    // set the specified bit to 1
    void setBit(const int x, const int y);

    // set the specified bit to 0
    void clearBit(const int x, const int y);

    char* linearPackData(void) const;

    // print the hex values of the cells of the raster
    //  not thread safe
    void print(const PosType max_x = ~0, const PosType max_y = ~0, const PosType min_x = 0, const PosType min_y = 0) const;
};