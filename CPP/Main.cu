#include "pch.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "env.hpp"
#include "Packer.cuh"
#include "Raster.hpp"
#include "RasterUtils.hpp"
#include "CudaMemManager.cuh"


// display certain device statistics to the console
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


__device__ inline int fromXY(const unsigned int x, const unsigned int y, const unsigned int width_stride){
    return y * width_stride + x;
}

template <int n_rounds>
__global__ void simpleCudaKernel(
    uint32_t* sheet_ptr, const unsigned int sheet_pitch_uint_32,
    uint32_t* output_ptr, const unsigned int output_pitch_uint_32,
    uint32_t* part_prt, const unsigned int part_pitch_uint32,
    const unsigned int sheet_width, const unsigned int sheet_height,
    const unsigned int part_width, const unsigned int part_height){
    // if(threadIdx.x == 0){
    //     printf("Hello from block %d %d, thread %d\n", blockIdx.x, blockIdx.y, threadIdx.x);
    // }

    // the x_block coordinate we are responsible for
    unsigned int my_x = blockIdx.x*blockDim.x + threadIdx.x;

    if(my_x > sheet_width){
        __syncthreads();
        return;
    }

    for(unsigned int round_y_offset = 0; round_y_offset < n_rounds; round_y_offset++){
        // miniumum possible region of the part that can overlap

        // the y_block coordinate we are responsible for
        unsigned int my_y = blockIdx.y*blockDim.y*n_rounds + round_y_offset;

        // the value of the sheet at the given cell
        const uint32_t sheet_value = sheet_ptr[fromXY(my_x, my_y, sheet_pitch_uint_32)];

        // TODO - remove
        //  sheet ptr should be const; this is just for testing
        // sheet_ptr[fromXY(my_x, my_y, sheet_pitch_uint_32)] = ~0;

        // TODO - check if these are really what we want
        const unsigned int min_part_x = min(my_x, part_width);
        const unsigned int min_part_y = min(my_y, part_height);
        const unsigned int max_part_x = ((part_width + my_x > sheet_width) ? (part_width + my_x - sheet_width) : 0);
        const unsigned int max_part_y = ((part_height + my_y > sheet_height) ? (part_height + my_y - sheet_height) : 0);

        // TODO - actually figure out
        for(unsigned int current_part_x = min_part_x; current_part_x <= max_part_x; current_part_x++){ // over x
            for(unsigned int current_part_y=min_part_y; current_part_y <= max_part_y; current_part_y++){ // over y
                const unsigned int current_part_x = 0;
                const unsigned int current_part_y = 0;

                const uint32_t a1 = sheet_ptr[fromXY(current_part_x, current_part_y, part_pitch_uint32)];
                const uint32_t a2 = ((current_part_y) ? (sheet_ptr[fromXY(current_part_x, current_part_y-1, part_pitch_uint32)]) : (0));

                uint32_t c = 0;

                #pragma unroll
                for(unsigned int i = 0; i < 32; i++){
                    const uint32_t t = ((a1 << i) & sheet_value) | ((a2 >> (32-i)) & sheet_value);
                    c |= (t ? (1 << i) : 0);
                }

                atomicOr(output_ptr + fromXY(current_part_x + my_x, current_part_y + my_y, output_pitch_uint_32), c);
            }
        }
    }

    __syncthreads();
}

// for each x_value find the lowest non-colliding y value
//  i.e. the first zero bit
__global__ void findBestPlacement(
    uint32_t* output_ptr, const unsigned int output_pitch_uint_32,
    const unsigned output_width, const unsigned int output_height_reg_32,
    uint32_t* storage
){
    // x-value of this worker
    const unsigned int my_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ uint32_t s[];

    s[threadIdx.x] = ~0;

    if(my_x >= output_width){
        // pass
    }else{
        unsigned int best_y = output_height_reg_32;
        for(unsigned int y_offset = 0; y_offset < output_height_reg_32; y_offset++){
            const unsigned int c = output_ptr[fromXY(my_x, y_offset, output_pitch_uint_32)];

            if(c == ~0){
                // all of the spots are collisions 
                continue;
            }

            constexpr uint8_t lut[] = {0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4};

            // there is a zero somewhere, so we can find it easily
            best_y = 31;
            #pragma unroll
            for(unsigned int i = 0; i < 32; i+=4){
                const uint32_t t = lut[(c & (0xF << i)) >> i];

                if(t == 4){
                    continue;
                }

                best_y = t + i;
                break;
            }
            
            s[threadIdx.x] = best_y + (y_offset * 32);
            break;
        }
    }
    __syncthreads();

    if(FAST_MOD_2(threadIdx.x) == 0){
        if(s[threadIdx.x] == ~0){
            s[threadIdx.x] = s[threadIdx.x+1];
        }
    }
    __syncthreads();
    if(FAST_MOD_4(threadIdx.x) == 0){
        if(s[threadIdx.x] == ~0){
            s[threadIdx.x] = s[threadIdx.x+2];
        }
    }
    __syncthreads();
    if(FAST_MOD_8(threadIdx.x) == 0){
        if(s[threadIdx.x] == ~0){
            s[threadIdx.x] = s[threadIdx.x+4];
        }
    }
    __syncthreads();
    if(FAST_MOD_16(threadIdx.x) == 0){
        if(s[threadIdx.x] == ~0){
            s[threadIdx.x] = s[threadIdx.x+8];
        }
    }
    __syncthreads();
    if(FAST_MOD_32(threadIdx.x) == 0){
        if(s[threadIdx.x] == ~0){
            s[threadIdx.x] = s[threadIdx.x+16];
        }
    }
    __syncthreads();
    if(FAST_MOD_64(threadIdx.x) == 0){
        if(s[threadIdx.x] == ~0){
            s[threadIdx.x] = s[threadIdx.x+32];
        }
    }
    __syncthreads();
    if(FAST_MOD_128(threadIdx.x) == 0){
        if(s[threadIdx.x] == ~0){
            s[threadIdx.x] = s[threadIdx.x+64];
        }
    }
    __syncthreads();

    // TODO - need to return the the y value and x value from the block
    if(threadIdx.x == 0){
        if(s[0] == ~0){
            storage[blockIdx.x] = s[128];
        }else{
            storage[blockIdx.x] = s[0];
        }
    }
}


// bake a part into the sheet at a given location
// TODO - incomplete
template <int n_rounds>
__global__ void bakePart(
    uint32_t* part_ptr, const unsigned int part_pitch_uint32,
    uint32_t* sheet_ptr, const unsigned int sheet_pitch_uint32,
    const unsigned int x, const unsigned int y ,
    const unsigned int part_width, const unsigned int part_height_uint32
){
    const unsigned int base_x = FAST_DIV_32(x)*32;
    const unsigned int base_y = FAST_DIV_32(y)*32;

    // the x_block coordinate we are responsible for
    unsigned int my_x = blockIdx.x*blockDim.x + threadIdx.x + base_x;

    if(my_x < x){
        // this thread is left of the part, and thus nothing to do
        __syncthreads();
        return;
    }

    if(my_x >= x + part_width){
        // this thread is right of the part, and thus nothing to do
        __syncthreads();
        return;
    }
    
    //y - base_y ensures we start at the approprite point
    // HOW TO DO ALIGN HERE AAAAAAA
    for(unsigned int round_y_offset = y - base_y; round_y_offset < n_rounds; round_y_offset++){

        // the y_block coordinate we are responsible for
        unsigned int my_y = blockIdx.y*blockDim.y*n_rounds + round_y_offset + base_y;

        if(my_y >= y + part_height){
            // we are below he
            continue;
        }

        uint32_t c = 0;

        atomicOr(sheet_ptr + fromXY(my_x + x, my_y + y, sheet_pitch_uint32), c);
    }

    __syncthreads();
}

void simple_cuda(const Raster& part){
    // lets compute some constants
    
    // width of the sheet in number of samples
    constexpr size_t sheet_width_samples = SHEET_WIDTH_INCH*SAMPLES_PER_INCH;

    // height of the sheet in number of samples
    constexpr size_t sheet_height_samples = SHEET_HEIGHT_INCH*SAMPLES_PER_INCH;

    // i just assume this is true
    //  and not sure what will break if it isnt
    //  next two statements for sure, but what else
    static_assert(sizeof(uint32_t) == 4);

    // height of the sheet in number of 32bit registers
    constexpr size_t sheet_height_reg_32 = CEIL_DIV_32(sheet_height_samples);

    // with of a row in memory in bytes
    constexpr size_t row_width_bytes = sizeof(uint32_t)*sheet_width_samples;

    // width of the ouput in number of samples
    const size_t output_width_samples = sheet_width_samples - part.getWidth() + 1;

    // height of the ouput in number of samples
    const size_t output_height_samples = sheet_height_samples - part.getHeight() + 1;

    // height of the output in number of 32bit registers
    const size_t output_height_reg_32 = CEIL_DIV_32(output_height_samples);
    
    // width of a row in number of bytes
    const size_t output_row_width_bytes = sizeof(uint32_t)*output_width_samples;

    // with of the part in number of samples
    const size_t part_width_samples = part.getWidth();

    // height of the part in number of samples
    const size_t part_height_samples = part.getHeight();

    // height of the part in number of 32bit registers
    const size_t part_height_reg_32 = CEIL_DIV_32(part_height_samples);

    // width of a row in number of bytes
    const size_t part_row_width_bytes = sizeof(uint32_t)*part_width_samples;
    
    // TODO - const qualify, const cast these six

    // pointer to __device__ memory allocated for storing sheet
    void* sheet_devptr;

    // value of the pitch for sheet device pointer
    //  subsequent rows are aligned to bus breaks (512 bytes?)
    size_t sheet_devpitch;

    // pointer to __device__ memory allocated for storing the part
    void* part_devptr = nullptr;

    // value of the pitch for part device pointer
    size_t part_devpitch;

    // pointer to __device__ memort allocated for storing the part
    void* output_devptr;

    // value of the pitch for output device pointer
    size_t output_devpitch;

    CUDACall(cudaMallocPitch(&sheet_devptr, &sheet_devpitch, row_width_bytes, sheet_height_reg_32));
    CUDACall(cudaMemset2D(sheet_devptr, sheet_devpitch, 0, row_width_bytes, sheet_height_reg_32));
    CUDACall(cudaMallocPitch(&output_devptr, &output_devpitch, output_row_width_bytes, output_height_reg_32));
    CUDACall(cudaMemset2D(output_devptr, output_devpitch, 0, output_row_width_bytes, output_height_reg_32));
    CUDACall(cudaMallocPitch(&part_devptr, &part_devpitch, part_row_width_bytes, part_height_reg_32));
    CUDACall(cudaMemset2D(part_devptr, part_devpitch, 0, part_row_width_bytes, part_height_reg_32));
    CUDACall(cudaDeviceSynchronize()); // force the prior operations to complete before proceeding

    // create the storage pointer
    const int num_reduce_storage_blocks = CEIL_DIV_256(output_width_samples);

    void* storage_devptr;
    CUDACall(cudaMalloc(&storage_devptr, sizeof(uint32_t)*num_reduce_storage_blocks));

    printf("Allocated arrays successfully.\n");

    // calculate block constants

    // number of threads per block (width)
    constexpr size_t block_width_thread = 32;

    // number of rounds that each thread will perform
    constexpr size_t block_height_rounds = 32;

    // calculate kernel constants

    // number of blocks in the kernel width
    constexpr size_t grid_width_blocks = FAST_DIV_32(sheet_width_samples);
    static_assert(block_width_thread == 32);

    constexpr size_t grid_height_blocks = FAST_DIV_32(sheet_height_reg_32);
    static_assert(block_height_rounds == 32);

    printf(
        "Blocks are %ld threads by %ld rounds, kernel is %ldx%ld blocks (%ldx%ld regs); "
        "Total %ld blocks.\n",
        block_width_thread, block_height_rounds,
        grid_width_blocks, grid_height_blocks, 
        sheet_width_samples, sheet_height_reg_32,
        grid_width_blocks * grid_height_blocks
    );

    // copy in the data for the part
    {
        const char* c = part.linearPackData();
        const size_t lcl_pitch = part.getWidth() * sizeof(uint32_t);
        CUDACall(cudaMemcpy2D(part_devptr, part_devpitch, c, lcl_pitch, part.getWidth() * sizeof(uint32_t), CEIL_DIV_32(part.getHeight()), cudaMemcpyHostToDevice))
        delete c;
    }

    // launch and run the kernel

    const dim3 block_shape = dim3(block_width_thread);
    const dim3 grid_shape = dim3(grid_width_blocks, grid_height_blocks);

    const auto start_time = std::chrono::high_resolution_clock::now();
    simpleCudaKernel<block_height_rounds><<<grid_shape, block_shape>>>(
        (uint32_t*)sheet_devptr, sheet_devpitch / sizeof(uint32_t),
        (uint32_t*)output_devptr, output_devpitch / sizeof(uint32_t),
        (uint32_t*)part_devptr, part_devpitch / sizeof(uint32_t),
        sheet_width_samples, sheet_height_samples,
        part.getWidth(), part.getHeight()
    );

    checkCudaError(__LINE__);
    
    // CUDACall(cudaDeviceSynchronize());
    findBestPlacement<<<num_reduce_storage_blocks, 256, 256*sizeof(uint32_t)>>>(
        (uint32_t*)output_devptr, output_devpitch / sizeof(uint32_t),
        output_width_samples, output_height_reg_32,
        (uint32_t*)storage_devptr
    );

    checkCudaError(__LINE__);
    
    CUDACall(cudaDeviceSynchronize());
    const auto end_time = std::chrono::high_resolution_clock::now();

    printf("Kernel took %ld us to run\n", 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count()
    );

    // TODO - these should actually be computed
    const unsigned int bake_pos_x = 0;
    // TODO - these should actually be computed
    const unsigned int bake_pos_y = 0;

    static_assert(block_height_rounds == 32);
    static_assert(block_width_thread == 32);

    const unsigned int num_blocks_x = CEIL_DIV_32(bake_pos_x + part.getWidth()) - FAST_DIV_32(bake_pos_x);
    const unsigned int num_blocks_y = CEIL_DIV_32(bake_pos_y + CEIL_DIV_32(part.getHeight())) - FAST_DIV_32(bake_pos_y);

    bakePart<block_height_rounds><<<dim3(num_blocks_x, num_blocks_y), dim3(block_width_thread)>>>(
        (uint32_t*)part_devptr, part_devpitch,
        (uint32_t*)sheet_devptr, sheet_devpitch,
        bake_pos_x, bake_pos_y,
        part.getWidth(),
    );
    checkCudaError(__LINE__);
    
    CUDACall(cudaDeviceSynchronize());
    printf("Finished baking part\n");

    // allocate local memory to store the output

    // number of local bytes to represent the sheet
    constexpr size_t local_sheet_bytes = row_width_bytes * sheet_height_reg_32;

    // number of local bytes to represent the output
    const size_t local_output_bytes = output_row_width_bytes * output_height_reg_32;
    
    // ptr to local memory for the sheet
    void* sheet_ptr = malloc(local_sheet_bytes);
    memset(sheet_ptr, 0b1010, local_sheet_bytes);
    const char* const sheet_ptr_c = (char*)sheet_ptr;

    // ptr to local memory for the output
    void* output_ptr = malloc(local_output_bytes);
    memset(output_ptr, 0b1010, local_output_bytes);
    const char* const output_ptr_c = (char*)output_ptr;

    // ptr to local memory for the storage
    void* storage_ptr = malloc(sizeof(uint32_t)*num_reduce_storage_blocks);
    const uint32_t* storage_ptr_uint32 = (uint32_t*)(storage_ptr);

    CUDACall(cudaMemcpy2D(sheet_ptr, row_width_bytes, sheet_devptr, sheet_devpitch, row_width_bytes, sheet_height_reg_32, cudaMemcpyDeviceToHost));
    CUDACall(cudaMemcpy2D(output_ptr, output_row_width_bytes, output_devptr, output_devpitch, output_row_width_bytes, output_height_reg_32, cudaMemcpyDeviceToHost));
    CUDACall(cudaMemcpy(storage_ptr, storage_devptr, sizeof(uint32_t)*num_reduce_storage_blocks, cudaMemcpyDeviceToHost));
    CUDACall(cudaDeviceSynchronize());

    printf("Synchronized success\n");
    printf("%ld\n", storage_ptr_uint32[0]);

    {
        size_t unset_memory_count = 0;
        size_t zeroed_memeory_count = 0;
        size_t correct_memory_count = 0;
        // start memory check
        for (size_t i = 0; i < local_sheet_bytes; i++)
        {
            if (sheet_ptr_c[i] == 0)
            {
                zeroed_memeory_count += 1;
            }
            else if (sheet_ptr_c[i] == ~0)
            {
                correct_memory_count += 1;
            }
            else
            {
                unset_memory_count += 1;
            }
        }

        printf("Memcheck %ldB: %ld unset, %ld zeroed, %ld correct\n",
               local_sheet_bytes, unset_memory_count,
               zeroed_memeory_count, correct_memory_count);
    }


    {
        size_t unset_memory_count = 0;
        size_t zeroed_memeory_count = 0;
        size_t correct_memory_count = 0;
        // start memory check
        for (size_t i = 0; i < local_output_bytes; i++)
        {
            if (output_ptr_c[i] == 0)
            {
                zeroed_memeory_count += 1;
            }
            else if (output_ptr_c[i] == ~0)
            {
                correct_memory_count += 1;
            }
            else
            {
                unset_memory_count += 1;
            }
        }

        printf("Memcheck %ldB: %ld unset, %ld zeroed, %ld correct\n",
               local_output_bytes, unset_memory_count,
               zeroed_memeory_count, correct_memory_count);
    }

    free(sheet_ptr);
    free(output_ptr);
    CUDACall(cudaFree(output_devptr));
    CUDACall(cudaFree(sheet_devptr));
    CUDACall(cudaFree(part_devptr));
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
    // hostPack_entry_cuda(r_vector);
    
    simple_cuda(r_vector[0]);

}