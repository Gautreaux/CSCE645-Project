#include "pch.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "env.hpp"
#include "Packer.cuh"
#include "Raster.hpp"
#include "RasterUtils.hpp"

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

// TODO - remove
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


// TODO - remove
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

    for(unsigned int round_y_offset = 0; round_y_offset < n_rounds; round_y_offset++){
        // miniumum possible region of the part that can overlap

        // the x_block coordinate we are responsible for
        unsigned int my_x = blockIdx.x*blockDim.x + threadIdx.x;

        // the y_block coordinate we are responsible for
        unsigned int my_y = blockIdx.y*blockDim.y*n_rounds + round_y_offset;

        // the value of the sheet at the given cell
        const uint32_t sheet_value = sheet_ptr[fromXY(my_x, my_y, sheet_pitch_uint_32)];

        // TODO - remove
        //  sheet ptr should be const; this is just for testing
        sheet_ptr[fromXY(my_x, my_y, sheet_pitch_uint_32)] = ~0;

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
    
    CUDACall(cudaDeviceSynchronize());
    const auto end_time = std::chrono::high_resolution_clock::now();

    printf("Kernel took %ld us to run\n", 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count()
    );

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

    CUDACall(cudaMemcpy2D(sheet_ptr, row_width_bytes, sheet_devptr, sheet_devpitch, row_width_bytes, sheet_height_reg_32, cudaMemcpyDeviceToHost));
    CUDACall(cudaMemcpy2D(output_ptr, output_row_width_bytes, output_devptr, output_devpitch, output_row_width_bytes, output_height_reg_32, cudaMemcpyDeviceToHost));
    CUDACall(cudaDeviceSynchronize());

    printf("Synchronized success\n");

    {
        size_t unset_memory_count = 0;
        size_t zeroed_memeory_count = 0;
        size_t correct_memory_count = 0;
        unsigned int ctr = 0;
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
        unsigned int ctr = 0;
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
    const dim3 gridShape = dim3(num_blocks_x, CEIL_DIV_32(space.getHeight()));
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
    // hostPack_entry_cuda(r_vector);
    
    simple_cuda(r_vector[0]);

}