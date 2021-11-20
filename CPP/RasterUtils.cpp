#include "RasterUtils.hpp"

// TODO - remove
#include <iostream>

namespace {
    Raster _readRasterUncompressed(const std::string& file_path){
        std::ifstream ifs(file_path);

        if(!ifs){
            throw std::runtime_error(" ReadRasterUncompressed Error opening file.\n");
        }

        std::string s;

        std::getline(ifs, s);

        auto i = s.find(" ");
        auto j = s.find(" ", i + 1);

        if(i == std::string::npos || j == std::string::npos){
            throw std::runtime_error("Error in file format, could not find number rows and columns");
        }

        const int num_rows = std::stoi(s.substr(0, i));
        const int num_cols = std::stoi(s.substr(i+1, j-i));

        // ignore remainder of row where there is the resolution tag
        //  probably something worth checking at some point but idk

        Raster r(num_cols, num_rows);

        unsigned int row_offset = 0;
        unsigned int column_offset = 0;
        unsigned int row_counter = 0; // int32 rows

        char c;
        while(ifs >> c){

            if(c != '0' && c !='1'){
                throw std::runtime_error("Unexpected char in input\n");
            }

            if(c == '1'){
                r[row_offset][column_offset] |= 1 << row_offset;
            }

            column_offset++;
            if(column_offset >= num_cols){
                column_offset = 0;
                row_offset++;

                if(FAST_MOD_32(row_offset) == 0){
                    row_counter++;
                    row_offset = 0;
                }
            }
        }

        if(row_offset + 32*row_counter != num_rows){
            // std::cout << num_rows << " " << row_offset << "+ 32 * " << row_counter << std::endl;
            throw std::runtime_error("Non-correct number of rows\n");
        }

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

    throw std::runtime_error("Not Implemented");
    // buffer[0] = r.getHeight();
    // buffer[0] = (buffer[0] << 16) | r.getWidth();

    // const auto sz = r.getLinearDataSize();

    // memcpy(buffer+1, r.getData(), sz);

    // return 1 + sz;
}


void bakeRaster_cpu(
    const Raster& part, Raster& space, 
    const PosType x, const PosType y,
    const uint8_t y_offset
){
    // these checks protect against postype becoming unsigned
    if(x < 0){
        throw std::invalid_argument("Illegal negative X value");
    }
    if(y < 0){
        throw std::invalid_argument("Illegal negative Y value");
    }

    // <0 checks will be optimized out, just present for completness
    if(y_offset >= 8 || y_offset < 0){
        throw std::invalid_argument("y_offset must be in range [0..7]");
    }

    // temporary item just until more advanced behavior is implemented
    if(y_offset != 0){
        throw std::invalid_argument("y_offset must be zero (for now)");
    }

    if(part.getWidth() + x > space.getWidth()){
        throw std::invalid_argument("Would overflow outside right of dest");
    }

    if(part.getHeight() + y > space.getHeight()){
        throw std::invalid_argument("Would overflow outside top of dest");
    }

    // the remainder should be guaranteed error free
    for(unsigned int j = 0; j < part.getHeight(); j++){
        for(unsigned int i=0; i < part.getWidth(); j++){
            space[j+y][i+x] |= part[j][i];
        }
    }
}


Raster buildPackMap_cpu(const Raster& space, const Raster& part){
    const PosType out_w = space.getWidth() - part.getWidth() + 1;
    const PosType out_h = space.getHeight() - part.getHeight() + 1;

    Raster output_raster(out_w, out_h);

    for(unsigned int j = 0; j < out_h; j++){
        for(unsigned int i = 0; i < out_w; i++){
            // TODO - need to do bitwise shifting and checking

            uint32_t temp_out = 0;
            for(unsigned int p_j = 0; p_j < part.getHeight(); p_j++){
                for(unsigned int p_i = 0; p_i < part.getWidth(); p_i++){
                    if(space[j + p_j][i + p_i] & part[p_j][p_i]){
                        // there is any collision
                        temp_out = 1;
                        p_j = part.getHeight()-1;
                        break;
                    }
                }
            }
            output_raster[j][i] = (temp_out ? 1 : 0);
        }
    }

    return output_raster;
}


void pickBestPosition_cpu(
    const Raster& space,
    PosType& out_x, PosType& out_y,
    uint8_t& out_y_offset
){
    // TODO - need to do bitwise operations
    out_y_offset = 0;

    for(unsigned int i = 0; i < space.getWidth(); i++){
        for(unsigned int j = 0; j < space.getHeight(); j++){
            const auto a = space[j][i];

            if(a == 0){
                // there is no collision
                out_x = i;
                out_y = j;
                return;
            }
        }
    }
}