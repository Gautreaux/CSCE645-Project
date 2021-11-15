#include "pch.h"

#include <iostream>

#include "Raster.cuh"
#include "Packer.cuh"


int main(const int argc, const char * const * const argv){
    std::cout << "Running: " << argv[0] << std::endl; 

    Position p_local;
    Position* p_dev;
    cudaMalloc(&p_dev, sizeof(Position));
    checkRaster<<<5, 1>>>(nullptr, nullptr, p_dev);
    cudaMemcpy(&p_local, p_dev, sizeof(Position), cudaMemcpyDeviceToHost);
    cudaFree(p_dev);

    std::cout << p_local.first << " " << p_local.second << std::endl;

}