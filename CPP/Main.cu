#include "pch.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "env.hpp"
#include "Packer.cuh"
#include "Raster.hpp"
#include "RasterUtils.hpp"
#include "CudaMemManager.cuh"

// some common constants

// number of vertical rounds per block
//  in kernels that suppourt vertical segmentation
constexpr unsigned int block_height_rounds = 32;

// standard block width in number of threads
constexpr unsigned int std_block_width = 32;

// expanded block width in number of threads
constexpr unsigned int expanded_block_with = 256;

inline unsigned int number_vertical_blocks(const int n_vertical_items){
    static_assert(block_height_rounds == 32);
    return CEIL_DIV_32(n_vertical_items);
}

inline unsigned int number_horizontal_blocks(const int n_horizontal_items){
    static_assert(std_block_width == 32);
    return CEIL_DIV_32(n_horizontal_items);
}

inline unsigned int expanded_number_horizontal_block(const int n_horizontal_items){
    static_assert(expanded_block_with == 256);
    return CEIL_DIV_256(n_horizontal_items);
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


__device__ inline int fromXY(const unsigned int x, const unsigned int y, const unsigned int width_stride){
    return y * width_stride + x;
}


// doesnt seem to produce the exact right results
template <int n_rounds>
__global__ void calculateCollisions(
    uint32_t const * const sheet_ptr, const unsigned int sheet_pitch_uint_32,
    uint32_t* const output_ptr, const unsigned int output_pitch_uint_32,
    uint32_t const * const part_ptr, const unsigned int part_pitch_uint32,
    const unsigned int output_width, const unsigned int output_height,
    const unsigned int part_width, const unsigned int part_height,
    const unsigned int sheet_height)
{
#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0){
        printf(
            "============== Calculate Collisions Entry ===================\n"
            "= %p %p %p\n"
            "= %u %u %u\n"
            "= %u %u %u %u\n"
            "= {%u %u %u} {%u %u %u}\n"
            "=============================================================\n",
            sheet_ptr, output_ptr, part_ptr,
            sheet_pitch_uint_32, output_pitch_uint_32, part_pitch_uint32,
            output_width, output_height, part_width, part_height,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z
        );
    }
#endif
    // each thread manages checking the whole part in a small area
    //  save some atomicOr operation
    // and allows for potentially alot of short circuiting if the 
    //  part collides quickly

    // the x_block coordinate we are responsible for
    //  in terms of collsion coordinates
    unsigned int my_x = blockIdx.x*blockDim.x + threadIdx.x;

    for(unsigned int round_y_offset = 0; round_y_offset < n_rounds; round_y_offset++){
        if(my_x >= output_width){
            __syncthreads();
            continue;
        }

        // the y_block coordinate we are responsible for
        //  in terms of collision coordinate
        const unsigned int my_y = blockIdx.y*n_rounds + round_y_offset;

        if(my_y >= output_height){
            break;
        }

        // since we are the only people resonsible for this collision
        //  we can just assume this is zero
        //  this may even save a memset later
        uint32_t out_value = 0;

        for(unsigned int x_offset = 0; x_offset < part_width; x_offset++){

            uint32_t part;
            uint32_t sheet_lower;
            uint32_t sheet_upper;

            for(unsigned int y_offset = 0; y_offset < CEIL_DIV_32(part_height); y_offset++){
                part = part_ptr[fromXY(my_x + x_offset, my_y + y_offset, part_pitch_uint32)];

                sheet_lower = sheet_upper;
                sheet_upper = ((my_y + y_offset * 32 < sheet_height) ? sheet_ptr[fromXY(my_x + x_offset, my_y + y_offset, sheet_pitch_uint_32)]
                                                                     : 0);

                if(sheet_lower == 0 && sheet_upper == 0){
                    continue;
                }
                if(sheet_lower == ~0 && sheet_upper == ~0){
                    out_value = ~0;
                    goto END_COLLIDE_DETECTED;
                }

                out_value |= (part & sheet_lower ? 1 : 0);

                for(unsigned int shift = 1; shift < 32; shift++){
                    out_value |= (
                        (((sheet_lower >> shift) | (sheet_upper << (32-shift))) & 
                         (part))
                        ? 1 : 0
                    ) << shift;
                }

                if(out_value == ~0){
                    goto END_COLLIDE_DETECTED;
                }
            }
        }

// Clayton will be so proud of me
END_COLLIDE_DETECTED:
        __syncthreads();

        // and then trivially write to output
        output_ptr[fromXY(my_x, my_y, output_pitch_uint_32)] = out_value;
    }
}

// for each x_value find the lowest non-colliding y value
//  i.e. the first zero bit
__global__ void findBestPlacement(
    uint32_t const * const output_ptr, const unsigned int output_pitch_uint_32,
    const unsigned output_width, const unsigned int output_height_reg_32,
    uint32_t * storage
){
#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0){
        printf(
            "============== Find Best Placement Entry ====================\n"
            "%p %p\n"
            "%u %u %u\n"
            "{%u %u %u} {%u %u %u}\n"
            "=============================================================\n",
            output_ptr, storage,
            output_pitch_uint_32, output_width, output_height_reg_32,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z
        );
    }
#endif

    // x-value of this worker
    const unsigned int my_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ uint32_t s[];

    // initalize shared memory
    s[threadIdx.x*2] = my_x;
    s[threadIdx.x*2+1] = ~0;

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
            
            s[threadIdx.x*2+1] = best_y + (y_offset * 32);
            break;
        }
    }
    __syncthreads();

    #pragma unroll
    for(unsigned int i = 1; i <= 8; i++){
        if((threadIdx.x & ((1 << i) - 1)) == 0){
            // when i = 1 take even threads
            // when i = 2 take every 4th thread
            // when i = 3 take every 8th thread

            // reduce, taking the left most item
            if(s[threadIdx.x*2+1] == ~0){
                // we did not find any valid position at this x value
                // so take the other position
                //  if its also invalid thats fine
                //  strictly speaking, its not worse
                s[threadIdx.x*2] = s[(threadIdx.x+(1 << (i - 1)))*2];
                s[threadIdx.x*2+1] = s[(threadIdx.x+(1 << (i - 1)))*2+1];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        storage[blockIdx.x*2] = s[0];
        storage[blockIdx.x*2+1] = s[1];
    }
}


// bake a part into the sheet at a given x,y location
// TODO - critical error in here somewhere when y != 0
template <int n_rounds>
__global__ void bakePart(
    uint32_t const * const part_ptr, const unsigned int part_pitch_uint32,
    uint32_t* const sheet_ptr, const unsigned int sheet_pitch_uint32,
    const unsigned int x, const unsigned int y,
    const unsigned int part_width, const unsigned int part_height
){

#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0){
        printf(
            "======================== Bake Part Entry ====================\n"
            "%p %p\n"
            "%u %u\n"
            "{%u %u} %u %u\n"
            "{%u %u %u}\n"
            "{%u %u %u}\n"
            "=============================================================\n",
            part_ptr, sheet_ptr,
            part_pitch_uint32, sheet_pitch_uint32, 
            x, y, part_width, part_height,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z
        );
    }
#endif

    const unsigned int base_x = CLAMP_32(x);
    const unsigned int base_y = CLAMP_32(y);
    const unsigned int part_height_r32 = CEIL_DIV_32(part_height);

    // the x_block coordinate we are responsible for
    //  relative to sheet coordinates
    const unsigned int sheet_x = blockIdx.x*blockDim.x + threadIdx.x + base_x;
    const unsigned int part_x = sheet_x - x; // may underflow and thats ok

    if(part_x >= part_width){
        // we are either left or right outside
        //  and thus have nothing to do
        __syncthreads();
        return;
    }

    // the above may alternatively be expressed as
    // if(sheet_x < x){
    //     // this thread is left of the part, and thus nothing to do
    //     __syncthreads();
    //     return;
    // }
    // if(sheet_x >= x + part_width){
    //     // this thread is right of the part, and thus nothing to do
    //     __syncthreads();
    //     return;
    // }

    uint32_t part_lower = 0;
    uint32_t part_upper = 0;

    const auto shift_amt = FAST_MOD_32(y);
    const auto shift_amt_comp = (32 - y);
    
    for(unsigned int round_y_offset = 0; round_y_offset < n_rounds; round_y_offset++){
        // the y_block coordinate we are responsible for
        //  relative to the sheet
        unsigned int sheet_y = blockIdx.y*blockDim.y*n_rounds + round_y_offset + base_y;

        unsigned int part_y = sheet_y - y; // this may underflow and thats a good thing

        // >= is critical for checking alignment
        if(part_y >= part_height_r32){
            // either underflowed so need another round
            // or overflowed and could break
            //  but its not that slow
            // alternatively: could explicitly check the two conditions
            continue;
        }

        part_lower = part_upper; // shift down previous
        part_upper = ((part_y < part_height_r32) ? part_ptr[fromXY(part_x, part_y, part_pitch_uint32)] : 0); // fetch new

        const uint32_t c = ((shift_amt == 0) ? (part_upper) : ((part_upper << shift_amt) & (part_lower >> shift_amt_comp)));

        const auto t = atomicOr(sheet_ptr + fromXY(sheet_x, sheet_y, sheet_pitch_uint32), c);

        if (t & c){
            printf("WARNING: Baking part produced a colliding cofiguration\n");
        }
    }

    __syncthreads();
}

void bakePartWrapper(
    const unsigned int bake_pos_x, const unsigned int bake_pos_y,
    const CudaMemManager2D &sheet, const CudaMemManager2D &part,
    const unsigned int part_true_h, const unsigned int part_true_w
)
{
    assert(part_true_w == part.width());

    const auto sheet_params = sheet.getDeviceParameters();
    const auto part_params = part.getDeviceParameters();

    // TODO - eval for correctness
    //  really really not convinced on these calculations
    const dim3 grid_shape_b = dim3(
        CEIL_DIV_32(bake_pos_x + part_true_w) - FAST_DIV_32(bake_pos_x),
        CEIL_DIV_32(bake_pos_y + CEIL_DIV_32(part_true_h)) - FAST_DIV_32(FAST_DIV_32(bake_pos_y))
    );
    const dim3 block_shape_b = dim3(std_block_width);

    const auto k_start_time = std::chrono::high_resolution_clock::now();

    bakePart<block_height_rounds><<<grid_shape_b, block_shape_b>>>(
        part_params.first, part_params.second,
        sheet_params.first, sheet_params.second,
        bake_pos_x, bake_pos_y,
        part_true_w, part_true_h
    );

    checkCudaError(__LINE__);

    CUDACall(cudaDeviceSynchronize());
    
    const auto k_end_time = std::chrono::high_resolution_clock::now();

    printf("Bake took (%ld us kernel)\n",
        std::chrono::duration_cast<std::chrono::microseconds>(k_end_time-k_start_time).count()
    );
}

void calculateCollisionsWrapper(
    const CudaMemManager2D& sheet, const CudaMemManager2D& part, 
    CudaMemManager2D& collision, CudaMemManager2D& reduce,
    const unsigned int sheet_true_h, const unsigned int sheet_true_w, 
    const unsigned int part_true_h, const unsigned int part_true_w,
    const unsigned int collision_true_h, const unsigned int collision_true_w,
    bool bake_result = false
){
    assert(sheet.width() == sheet_true_w);
    assert(part.width() == part_true_w);
    assert(collision.width() == collision_true_w);

    printf("Starting a collision calculation\n");
    const dim3 block_shape_c = dim3(std_block_width);
    const dim3 grid_shape_c = dim3(number_horizontal_blocks(collision.width()), number_vertical_blocks(collision.height()));

    const dim3 block_shape_r = dim3(expanded_block_with);
    const dim3 grid_shape_r = dim3(expanded_number_horizontal_block(collision.width()));

    const auto sheet_params = sheet.getDeviceParameters();
    const auto part_params = part.getDeviceParameters();
    const auto collision_params = collision.getDeviceParameters();
    const auto reduce_params = reduce.getDeviceParameters();

#ifdef DEBUG
    printf("!> Collision Config: {%u} {%u %u %u} {%u %u %u}\n", 
        block_height_rounds,
        grid_shape_c.x, grid_shape_c.y, grid_shape_c.z,
        block_shape_c.x, block_shape_c.y, block_shape_c.z
    );
    printf("!> Collision Args: <ptr> %lu <ptr> %lu <prt> %lu {%u %u %u %u}\n",
        sheet_params.second, collision_params.second, part_params.second,
        sheet_true_w, sheet_true_h, part_true_w, part_true_h
    );
    printf("!> Reduce Config: {%u %u %u} {%u %u %u} {%lu}\n",
        grid_shape_r.x, grid_shape_r.y, grid_shape_r.z,
        block_shape_r.x, block_shape_r.y, block_shape_r.z,
        sizeof(uint32_t)*2*expanded_block_with
    );
    printf("!> Reduce Args: <ptr> %lu %u %lu <ptr>\n", 
        collision_params.second, collision_true_w, collision.height()
    );
#endif

    const auto k_start_time = std::chrono::high_resolution_clock::now();

    calculateCollisions<block_height_rounds><<<grid_shape_c, block_shape_c>>>(
        sheet_params.first, sheet_params.second,
        collision_params.first, collision_params.second,
        part_params.first, part_params.second,
        collision_true_w, collision_true_h,
        part_true_w, part_true_h,
        sheet_true_h
    );

    checkCudaError(__LINE__);

    findBestPlacement<<<grid_shape_r, block_shape_r, sizeof(uint32_t)*2*expanded_block_with>>>(
        collision_params.first, collision_params.second,
        collision_true_w, collision.height(),
        reduce_params.first
    );

    CUDACall(cudaDeviceSynchronize());

    const auto k_end_time = std::chrono::high_resolution_clock::now();
    
    reduce.Pull();

#ifdef DEBUG
    printf("Reduce Results: ");
    for(unsigned int i = 0; i < grid_shape_r.x; i++){
        printf("{%u %u} ", reduce.at(i*2,0), reduce.at(i*2+1, 0));
    }
    printf("\n");
#endif

    unsigned int best_x = ~0;
    unsigned int best_y = ~0;

    for(unsigned int i = 0; i < grid_shape_r.x; i++){
        best_x = reduce.at(i*2,0);
        best_y = reduce.at(i*2+1,0);
        if(best_y != ~0){
            break;
        }
    }

    const auto end_time = std::chrono::high_resolution_clock::now();

    printf("Collsion calc returned %d %d in %ld us (%ld us kernel)\n",
        best_x, best_y,
        std::chrono::duration_cast<std::chrono::microseconds>(end_time-k_start_time).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(k_end_time-k_start_time).count()
    );

    if(bake_result){
        bakePartWrapper(
            best_x, best_y,
            sheet, part,
            part_true_h, part_true_w
        );
    }
}


void simple_cuda(const Raster& part){
    // lets compute some constants
    
    // width of the sheet in number of samples
    constexpr size_t sheet_width_samples = SHEET_WIDTH_INCH*SAMPLES_PER_INCH;

    // height of the sheet in number of samples
    constexpr size_t sheet_height_samples = SHEET_HEIGHT_INCH*SAMPLES_PER_INCH;

    // width of the ouput in number of samples
    const size_t output_width_samples = sheet_width_samples - part.getWidth() + 1;

    // height of the ouput in number of samples
    const size_t output_height_samples = sheet_height_samples - part.getHeight() + 1;

    // with of the part in number of samples
    const size_t part_width_samples = part.getWidth();

    // height of the part in number of samples
    const size_t part_height_samples = part.getHeight();



    // i just assume this is true
    //  and not sure what will break if it isnt
    //  next two statements for sure, but what else
    static_assert(sizeof(uint32_t) == 4);

    const bool b_sync = true;

    CudaMemManager2D sheet_mem(sheet_width_samples, CEIL_DIV_32(sheet_height_samples), b_sync);
    CudaMemManager2D part_mem(part, b_sync);
    printf("Collisions: ");
    CudaMemManager2D collisions_mem(output_width_samples, CEIL_DIV_32(output_height_samples), b_sync);
    printf("Reduce: ");
    CudaMemManager2D reduce_mem(expanded_number_horizontal_block(output_width_samples)*2, 1, b_sync);

    CudaMemManager2D::Sync();

    printf("Done initilizing Managed 2D Memory\n");

    calculateCollisionsWrapper(
        sheet_mem, part_mem, collisions_mem, reduce_mem,
        sheet_height_samples, sheet_width_samples, 
        part_height_samples, part_width_samples,
        output_height_samples, output_width_samples,
        true
    );

    sheet_mem.Pull();

    for(unsigned int i = 0; i < 8; i++){
        printf("%08X -> %08X ... %08X -> %08X  %08X -> %08X ... %08X -> %08X  00000000 -> %08X\n", 
            part_mem.at(0,i), sheet_mem.at(0,i),
            part_mem.at(42,i), sheet_mem.at(42,i),
            part_mem.at(43,i), sheet_mem.at(43,i),
            part_mem.at(2267,i), sheet_mem.at(2267,i),
            sheet_mem.at(2268,i)
        );
    }
    printf("....\n");
    {
        unsigned int i = 30;
        printf("%08X -> %08X ... %08X -> %08X  %08X -> %08X ... %08X -> %08X  00000000 -> %08X\n", 
            part_mem.at(0,i), sheet_mem.at(0,i),
            part_mem.at(42,i), sheet_mem.at(42,i),
            part_mem.at(43,i), sheet_mem.at(43,i),
            part_mem.at(2267,i), sheet_mem.at(2267,i),
            sheet_mem.at(2268,i)
        );
    }

    calculateCollisionsWrapper(
        sheet_mem, part_mem, collisions_mem, reduce_mem,
        sheet_height_samples, sheet_width_samples, 
        part_height_samples, part_width_samples,
        output_height_samples, output_width_samples,
        true
    );

    calculateCollisionsWrapper(
        sheet_mem, part_mem, collisions_mem, reduce_mem,
        sheet_height_samples, sheet_width_samples, 
        part_height_samples, part_width_samples,
        output_height_samples, output_width_samples,
        false
    );

    // for(unsigned int i = 0; i < 8; i++){
    //     for(unsigned int j = 0; j < 4; j++){
    //         printf("%08X ", collisions_mem.at(j,i));
    //     }
    //     printf("\n");
    // }

    // bakePartWrapper(
    //     0, 32,
    //     sheet_mem, part_mem,
    //     part_height_samples, part_width_samples
    // );

    // bakePartWrapper(
    //     0, 64,
    //     sheet_mem, part_mem,
    //     part_height_samples, part_width_samples
    // );

    return;
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

    r_vector[0].print(8, 385);
    
    simple_cuda(r_vector[0]);

}