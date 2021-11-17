#include "RasterUtils.hpp"

//TODO - remove this
#include <iostream>

namespace {
    Raster _readRasterUncompressed(const std::string& file_path){
        std::ifstream ifs(file_path);

        if(!ifs){
            throw std::runtime_error("Error opening file.\n");
        }

        std::string s;

        std::getline(ifs, s);

        auto i = s.find(" ");
        auto j = s.find(" ", i + 1);

        if(i == std::string::npos || j == std::string::npos){
            throw std::runtime_error("Error in file format");
        }

        const int num_rows = std::stoi(s.substr(0, i));
        const int num_cols = std::stoi(s.substr(i+1, j-i));

        const int num_int32_rows = (num_rows >> 5) + ((num_rows & 0b11111) ? 1 : 0);
        std::cout << num_rows << " (" << num_int32_rows << ") " << num_cols << std::endl;

        Raster r(num_cols, num_rows);

        uint32_t* const data = r.getData();

        // TODO - actually load the data

        return r;
    }
}

Raster readRaster(const std::string& file_path, const bool is_compressed){
    if(is_compressed){
        throw std::runtime_error("Cannot yet proces compressed files\n");
    }else{
        return _readRasterUncompressed(file_path);
    }
}

int packRaster(const Raster& r, int32_t * const buffer){
    static_assert((2*sizeof(PosType)) <= 4);

    buffer[0] = r.getHeight();
    buffer[0] = (buffer[0] << 16) | r.getWidth();

    const auto sz = r.getLinearDataSize();

    memcpy(buffer+1, r.getData(), sz);

    return 1 + sz;
}