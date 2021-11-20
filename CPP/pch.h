#pragma once

// this macro controls if items are compiled host and device
//  or if they are removed to allow for normal compilation with g++
#ifdef __NVCC__
#define NVCC_HD __host__ __device__
#else
#define NVCC_HD
#endif

// (x % 32)
#define FAST_MOD_32(x) (x & 0b11111)

// ceil(x/32)
#define CEIL_DIV_32(x) ((x >> 5) + (FAST_MOD_32(x) ? 1 : 0)) 