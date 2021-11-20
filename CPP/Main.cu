#include "pch.h"

#include <iostream>

#include "Raster.cuh"
#include "Packer.cuh"
#include "RasterUtils.hpp"

#include <vector>

#define SHEET_HEIGHT_U32_CELLS 128
#define SHEET_WIDTH_CELLS (48*170)

// packing of vector of rasters into linear memory and copy to device
//  returns a device pointer of where the 
int32_t* packPrep(const std::vector<Raster>& v){
    if(v.size() == 0){
        throw std::runtime_error("Error raster vector size is 0; nothing to pack\n");
    }

    size_t mem_required = 1; // plus one for the null character
    for(auto& r : v){
        mem_required += r.getLinearDataSize() + 1;
    }

    int32_t* host_buffer = (int32_t*)malloc(mem_required*sizeof(int32_t));

    if(host_buffer == nullptr){
        std::cout << "Memory Required: " << mem_required << std::endl;
        throw std::runtime_error("Could not allocate enough local memory");
    }

    int32_t* travler = host_buffer;

    for(auto& r : v){
        auto k = packRaster(r, travler);
        travler += k;
    }

    *travler = 0; // explicitly set terminating null character

    // now for the CUDA fun-ness
    int32_t* device_buffer; // buffer on the cuda device

    // TODO - proper error checking on CUDA calls
    cudaMalloc(&device_buffer, mem_required*sizeof(int32_t));
    cudaMemcpy(device_buffer, host_buffer, sizeof(int32_t)*mem_required, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    free(host_buffer);

    return device_buffer;
}

// allocate a sheet to work with
//  for now, a 24x48 sheet @170 pts per inch
//      but conceptually could be more in the future
//      could also do multi-sheet by concating them and
//       adding rows/columns of 1s to delimit seams
int32_t* sheetPrep(void){
    const auto n_bytes = SHEET_HEIGHT_U32_CELLS * 4 * SHEET_WIDTH_CELLS;
    
    int32_t* device_buffer;

    //TODO - proper error checking on CUDA call
    cudaMalloc(&device_buffer, n_bytes);
    cudaDeviceSynchronize();

    return device_buffer;
}

// allocate a buffer into which to place the output
int32_t* outputPrep(const size_t number_items){
    const size_t n_bytes = 2*sizeof(PosType)*number_items;

    int32_t* device_buffer;

    //TODO - proper error checking on CUDA call
    cudaMalloc(&device_buffer, n_bytes);
    cudaMemset(device_buffer, 0, n_bytes);
    cudaDeviceSynchronize();

    return device_buffer;
}

// do post pack cleanup
void cleanupDevice(
    int32_t* const raster_buffer, 
    int32_t* const sheet_buffer,
    int32_t* const output_buffer){
    
    //TODO - proper error checking on CUDA calls
    cudaFree(raster_buffer);
    cudaFree(sheet_buffer);
    cudaFree(output_buffer);
    cudaDeviceSynchronize();
}

// device entry for pack function
//  placeholder, will be removed eventually
__global__ void devicePack_entry(
    int32_t * raster_buffer,
    int32_t *const sheet_buffer,
    int32_t * output_buffer)
{
    while(*raster_buffer != 0){
        // this loop packs a single element
        int16_t width = raster_buffer[0] & (0x0000FFFF);
        int16_t height = (raster_buffer[0] & 0xFFFF0000) >> 16;
        
        int16_t n_rounds_horz = SHEET_WIDTH_CELLS;
        int16_t
    }
}

// host entry for the pack function
void hostPack_entry(const std::vector<Raster>& to_pack){
    std::cout << "Beginning Packing Prep" << std::endl;
    // may need to unpack into one function and synchronize
    //  if wierd behavior is observed
    const auto raster_buffer = packPrep(to_pack);
    const auto sheet_buffer = sheetPrep();
    const auto output_buffer = outputPrep(to_pack.size());

    std::cout << "Beginning Pack Kernel" << std::endl;
    // TODO - invoke kernel
    cudaDeviceSynchronize();

    std::cout << "Ended Pack Kernel" << std::endl;
    cleanupDevice(raster_buffer, sheet_buffer, output_buffer);
}

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

    hostPack_entry(r_vector);

}