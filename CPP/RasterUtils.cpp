#include "RasterUtils.hpp"


namespace {
    Raster _readRasterUncompressed(const std::string& file_path){
        throw std::runtime_error("Not Implemented");
        // std::ifstream ifs(file_path);

        // if(!ifs){
        //     throw std::runtime_error("Error opening file.\n");
        // }

        // std::string s;

        // std::getline(ifs, s);

        // auto i = s.find(" ");
        // auto j = s.find(" ", i + 1);

        // if(i == std::string::npos || j == std::string::npos){
        //     throw std::runtime_error("Error in file format");
        // }

        // const int num_rows = std::stoi(s.substr(0, i));
        // const int num_cols = std::stoi(s.substr(i+1, j-i));

        // // const int num_int32_rows = (num_rows >> 5) + ((num_rows & 0b11111) ? 1 : 0);
        // // std::cout << num_rows << " (" << num_int32_rows << ") " << num_cols << std::endl;

        // Raster r(num_cols, num_rows);

        // uint32_t* const data = r.getData();

        // unsigned int row_offset = 0;
        // unsigned int column_offset = 0;
        // unsigned int row_counter = 0; // int32 rows

        // char c;
        // while(ifs >> c){

        //     if(c != '0' && c !='1'){
        //         throw std::runtime_error("Unexpected char in input\n");
        //     }

        //     if(c == '1'){
        //         data[row_counter*num_cols+column_offset] |= 1 << row_offset;
        //     }

        //     column_offset++;
        //     if(column_offset >= num_cols){
        //         column_offset = 0;
        //         row_offset++;

        //         if((row_offset & (0b11111)) == 0){
        //             row_counter++;
        //             row_offset = 0;
        //         }
        //     }
        // }


        // if(row_offset + 32*row_counter != num_rows){
        //     // std::cout << num_rows << " " << row_offset << "+ 32 * " << row_counter << std::endl;
        //     throw std::runtime_error("Non-correct number of rows\n");
        // }

        // return r;
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