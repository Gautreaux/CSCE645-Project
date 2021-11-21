#include "pch.h"

#include <iostream>

#include "env.hpp"
#include "Packer.cuh"
#include "Raster.hpp"
#include "RasterUtils.hpp"

#include <vector>

void displayCUDAdeviceStats(void){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Detected " << deviceCount << " devices" << std::endl;

    if(deviceCount <= 0){
        std::cout << "Error no device detected" << std::endl;
        std::exit(2);
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
        printf("  %s\n", deviceProp.name);
        printf("  %dkHz, %d threads per block, %d threads per mp, %d mp on device, %d warp size\n",
            deviceProp.clockRate, deviceProp.maxThreadsPerBlock, 
            deviceProp.maxThreadsPerMultiProcessor,
            deviceProp.multiProcessorCount, deviceProp.warpSize);
        // printf("  %d i32 reg per block, %d i32 reg per mp\n", deviceProp.regsPerBlock, deviceProp.regsPerMultiprocessor);

        printf("  %lu MB device global mem, %lu (%lu opt) B shared mem per block\n", 
            (deviceProp.totalGlobalMem) >> 20, 
            (deviceProp.sharedMemPerBlock),
            (deviceProp.sharedMemPerBlockOptin));

        if(deviceProp.kernelExecTimeoutEnabled){
            printf("  [CRITICAL] Kernel Timeout is enabled (no idea what it is lol)\n");
        }
    }
}

void hostPack_entry_cpu(const std::vector<Raster>& to_pack){
    Raster space(SHEET_WIDTH_INCH*SAMPLES_PER_INCH, SHEET_HEIGHT_INCH*SAMPLES_PER_INCH);

    for(unsigned int i =0; i < to_pack.size(); i++){
        const Raster& this_part = to_pack[i];

        std::cout << "S";
        std::flush(std::cout);

        const Raster& possible_locations = buildPackMap_cpu(space, this_part);

        std::cout << ".M";
        std::flush(std::cout);

        PosType x_place, y_place;
        uint8_t y_out_offset;

        pickBestPosition_cpu(possible_locations, x_place, y_place, y_out_offset);
        std::cout << ".P-";
        std::flush(std::cout);
        bakeRaster_cpu(this_part, space, x_place, y_place, y_out_offset);
        std::cout << ".DONE  ";
        std::flush(std::cout);

        std::cout << "Packer placed raster." << i << " at " << x_place 
            << " " << y_place << "+" << y_out_offset << std::endl;

    }
}

__global__ void buildCollisionMap(
    const uint32_t* const part, const uint32_t* const space, 
    uint32_t* const out_array, const PosType part_width,
    const PosType part_height)
{
    uint32_t my_x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t my_y = blockDim.y * blockIdx.y;

    out_array[my_x] = 5;

    // TODO - do something here;
}

void hostPack_entry_cuda(const std::vector<Raster>& to_pack){
    // the Raster representing the sheet
    Raster space(SHEET_WIDTH_INCH*SAMPLES_PER_INCH, SHEET_HEIGHT_INCH*SAMPLES_PER_INCH);

    void* sheet_on_device_ptr;
    size_t pitch_device;

    // TODO - error checking
    cudaMallocPitch(&sheet_on_device_ptr, &pitch_device, sizeof(uint32_t)*space.getWidth(), CEIL_DIV_32(space.getHeight()));
    cudaMemset2D(sheet_on_device_ptr, pitch_device, 0, sizeof(uint32_t)*space.getWidth(), CEIL_DIV_32(space.getHeight()));
    cudaDeviceSynchronize();

    const Raster& r_1 = to_pack[0];
    char* const r_1_linearized = r_1.linearPackData();

    void* part_on_device_ptr;
    size_t pitch_part;

    // TODO - error checking
    cudaMallocPitch(&part_on_device_ptr, &pitch_part, sizeof(uint32_t)*r_1.getWidth(), CEIL_DIV_32(r_1.getHeight()));
    cudaMemcpy2D(part_on_device_ptr, pitch_part, r_1_linearized, sizeof(uint32_t)*r_1.getWidth(), sizeof(uint32_t)*r_1.getWidth(), r_1.getHeight(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    const size_t out_width = space.getWidth() - r_1.getWidth() + 1;
    const size_t out_height = CEIL_DIV_32(space.getHeight()) - CEIL_DIV_32(r_1.getHeight()) + 1;
    
    void* out_on_device_ptr;
    size_t pitch_out;
    
    // TODO - error checking
    cudaMallocPitch(&out_on_device_ptr, &pitch_out, sizeof(uint32_t)*out_width, out_height);
    cudaMemset2D(out_on_device_ptr, pitch_out, 0, out_width*sizeof(uint32_t), out_height);
    cudaDeviceSynchronize();

    const int num_blocks_x = CEIL_DIV_32(space.getWidth());
    const dim3 gridShape = dim3(num_blocks_x, 8);
    buildCollisionMap<<<gridShape, 32>>>(
        (uint32_t*)part_on_device_ptr, (uint32_t*)sheet_on_device_ptr, 
        (uint32_t*)out_on_device_ptr, r_1.getWidth(), CEIL_DIV_32(r_1.getHeight()));
    cudaDeviceSynchronize();
    std::cout << "All done" << std::endl;
}

int main(const int argc, const char * const * const argv){
    // TODO - endianness checks on entry

    std::cout << "Running: " << argv[0] << std::endl; 

    displayCUDAdeviceStats();

    // For now, assume that we want the first device always
    // Set device 0 as current
    cudaSetDevice(0);


    // Position p_local;
    // Position* p_dev;
    // cudaMalloc(&p_dev, sizeof(Position));
    // checkRaster<<<5, 1>>>(nullptr, nullptr, p_dev);
    // cudaMemcpy(&p_local, p_dev, sizeof(Position), cudaMemcpyDeviceToHost);
    // cudaFree(p_dev);
    // std::cout << p_local.first << " " << p_local.second << std::endl;

    std::vector<std::string> filepaths_vector = {
        "../SampleRasters/basebot.raster",
        "../SampleRasters/blocker.raster",
        "../SampleRasters/Part7.raster",
        "../SampleRasters/vert1.raster",
        "../SampleRasters/vert2.raster",
        };

    std::vector<Raster> r_vector;

    for (const auto& fp : filepaths_vector){
        r_vector.emplace_back(readRaster(fp));
    }

    std::cout << "Loaded test rasters, startingpack." << std::endl;

    // hostPack_entry(r_vector);
    // hostPack_entry_cpu(r_vector);
    hostPack_entry_cuda(r_vector);
    

}