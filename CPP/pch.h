#pragma once

// this macro controls if items are compiled host and device
//  or if they are removed to allow for normal compilation with g++
#ifdef __NVCC__
#define NVCC_HD __host__ __device__
#else
#define NVCC_HD
#endif