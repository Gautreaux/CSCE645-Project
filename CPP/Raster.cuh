#pragma once
#include "pch.h"

#include <cstdint>
#include <utility>


using PosType = uint16_t;
using Position = std::pair<PosType, PosType>;

class Raster{
protected:
    PosType width;
    PosType height;
    uint32_t** data;
private:

};