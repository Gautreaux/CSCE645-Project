#include "pch.h"

#include <cinttypes>
#include <stdio.h>

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

enum ManagedMallocLocation{
    NONE   = 0,
    HOST   = 0b01,
    DEVICE = 0b10,
    BOTH   = 0b11
};

template <class InterpretedType>
class CudaMemManager{
    InterpretedType* host;
    InterpretedType* device;
    size_t host_pitch;
    size_t device_pitch;
    size_t width;
    size_t height;
    bool host_aligned;
    bool dev_aligned;

public:
    CudaMemManager(void): 
        host(nullptr), device(nullptr), host_pitch(0), device_pitch(0), 
        width(0), height(0), dev_aligned(false), host_aligned(false)
    {}
    
    ~CudaMemManager(void){
        if(host){
            free(host);
        }
        if(device){
            CUDACall(cudaFree(device));
            CUDACall(cudaDeviceSynchronize());
        }
    }

    // managed malloc of a single dimensional array, no alignment
    void MangedMalloc(const unsigned int num_inst, const ManagedMallocLocation location)
    {
        if (num_inst == 0)
        {
            return;
        }

        const size_t raw_bytes = num_inst * sizeof(InterpretedType);

        if (location & ManagedMallocLocation::HOST)
        {
            if (host)
            {
                throw "Cannot realloc host memory (yet)";
            }
            host = (InterpretedType*)malloc(raw_bytes);
            // host_pitch = ;
        }

        if (location & ManagedMallocLocation::DEVICE)
        {
            if (device)
            {
                throw "Cannot realloc host memory (yet)";
            }

            void* t_ptr;

            CUDACall(cudaMalloc(&t_ptr, raw_bytes))

            device = (InterpretedType*)t_ptr;
            // device_pitch = t_ptr;
        }
    }

    // TODO finish this class
};