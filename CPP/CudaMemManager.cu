#include "CudaMemManager.cuh"

CudaMemManager2D::CudaMemManager2D(const size_t w, const size_t h, const bool sync_cuda_block): 
    width_samples(w), height_samples(h), device_stride(0), raw_width_b(sizeof(InterpretedType)*width_samples), host_stride(raw_width_b),
    host_ptr(malloc(host_stride*height_samples)), device_ptr(nullptr)
{
    void* dev_ptr_lcl;
    size_t dev_stride_lcl;


    CUDACall(cudaMallocPitch(&dev_ptr_lcl, &dev_stride_lcl, raw_width_b, height_samples));

    printf("Alloc ptr (dev): %p %lu\n", dev_ptr_lcl, dev_stride_lcl);

    const_cast<void*&>(device_ptr) = dev_ptr_lcl;
    const_cast<size_t&>(device_stride) = dev_stride_lcl;


    this->Memset(0, false);

    if(sync_cuda_block){
        this->Sync();
    }

    // not sure how important this is, but just in case
    assert(device_stride % sizeof(InterpretedType) == 0);
    assert(host_stride % sizeof(InterpretedType) == 0);
}

CudaMemManager2D::CudaMemManager2D(const Raster& r, bool sync_cuda_block) : 
    CudaMemManager2D(r.getWidth(), CEIL_DIV_32(r.getHeight()), false)
{
    assert(host_stride*height_samples <= r.getLinearDataSize()*sizeof(uint32_t));

    r.linearPackData((char*)host_ptr);

    this->Push(false);

    if(sync_cuda_block){
        this->Sync();
    }
}

CudaMemManager2D::~CudaMemManager2D(void){
    free(host_ptr);
    CUDACall(cudaFree(device_ptr));
    CUDACall(cudaDeviceSynchronize());
}

void CudaMemManager2D::Memset(const int c, const bool sync_cuda_block){
    memset(host_ptr, c, host_stride*height_samples);
    CUDACall(cudaMemset2D(device_ptr, device_stride, c, raw_width_b, height_samples))
    
    if(sync_cuda_block){
        this->Sync();
    }
}

void CudaMemManager2D::Push(const bool sync_cuda_block) const {
    printf("Pushing data\n");

    CUDACall(cudaMemcpy2D(
        device_ptr, device_stride, 
        host_ptr, host_stride, 
        raw_width_b, height_samples,
        cudaMemcpyHostToDevice
    ));

    if(sync_cuda_block){
        this->Sync();
    }
}

void CudaMemManager2D::Pull(const bool sync_cuda_block){
    printf("Pulling data\n");

    CUDACall(cudaMemcpy2D(
        host_ptr, host_stride, 
        device_ptr, device_stride, 
        raw_width_b, height_samples,
        cudaMemcpyDeviceToHost
    ));

    if(sync_cuda_block){
        this->Sync();
    }
}

std::pair<InterpretedType*, size_t> CudaMemManager2D::getDeviceParameters(void) const {
    return {(InterpretedType*)device_ptr, device_stride / sizeof(InterpretedType)};
}

const InterpretedType& CudaMemManager2D::at(const int x, const int y) const{
    assert(x >= 0 && x < width_samples);
    assert(y >= 0 && y < height_samples);

    return *(InterpretedType*)(((char*)host_ptr) + (y * host_stride + x*sizeof(InterpretedType)));
}