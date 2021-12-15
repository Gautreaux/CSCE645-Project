#include "pch.h"

#include <cinttypes>
#include <stdio.h>

#include <utility>
#include <cassert>

#include "Raster.hpp"

#define CUDACall(x) {x; checkCudaError(__LINE__);}

// check if a cuda error occrred and exit if so
inline void checkCudaError(const unsigned int line){
    cudaError_t last_err;
    if((last_err = cudaGetLastError()) != 0){
        printf("[%s:%d] Cuda error %d: %s\n", 
        __FILE__, line, last_err, cudaGetErrorName(last_err));
        exit(1);
    }
}


using InterpretedType = uint32_t;

static_assert(sizeof(InterpretedType) == 4);

class CudaMemManager2D{
    // Could never get these to work
    // // are the devices out of sync
    // bool is_stale;
    // // is the host or the device the leading object
    // bool host_ahead;

    // the width of a row in number of samples
    const size_t width_samples;

    // the height in number of samples
    const size_t height_samples;

    // the width in bytes of a row with no padding
    const size_t raw_width_b;

    // the width of a device row in bytes, including padding
    const size_t device_stride;

    // the width of a host row in bytes including padding
    const size_t host_stride;
    
    void* const  host_ptr;

    void* const device_ptr;

public:
    CudaMemManager2D(void) = delete;
    CudaMemManager2D(const CudaMemManager2D&) = delete;
    CudaMemManager2D(CudaMemManager2D&&) = delete;

    explicit CudaMemManager2D(const size_t w, const size_t h, const bool sync_cuda_block = true);
    
    explicit CudaMemManager2D(const Raster& r, bool sync_cuda_block = false);

    ~CudaMemManager2D(void);

    inline static void Sync(void){
        CUDACall(cudaDeviceSynchronize());
    }

    // memset on both the host and device
    void Memset(const int c, const bool sync_cuda_block = true);

    // push the local buffer to the device 
    void Push(const bool sync_cuda_block = true) const;

    // pull the device contents to the local buffer
    void Pull(const bool sync_cuda_block = true);


    inline size_t height(void) const {
        return height_samples;
    }

    inline size_t width(void) const {
        return width_samples;
    }

    inline size_t getHostStride(void) const {
        return host_stride;
    }

    inline size_t getDeviceStride(void) const {
        return device_stride;
    }

    inline size_t getRawWidthBytes(void) const {
        return raw_width_b;
    }

    // return a pair of relevant info
    // **IN TERMS OF InterpretedType**
    std::pair<InterpretedType*, size_t> getDeviceParameters(void) const;

    const InterpretedType& at(const int x, const int y) const;

    inline void* gethostPtr_unsafe(void) const {
        return host_ptr;
    }

    inline void* getDevPtr_unsafe(void) const {
        return device_ptr;
    }
};