#pragma once

// this macro controls if items are compiled host and device
//  or if they are removed to allow for normal compilation with g++
#ifdef __NVCC__
#define NVCC_HD __host__ __device__
#else
#define NVCC_HD
#endif

// (x % 32)
#define FAST_MOD_2(x)   (x & 0b1)
#define FAST_MOD_4(x)   (x & 0b11)
#define FAST_MOD_8(x)   (x & 0b111)
#define FAST_MOD_16(x)  (x & 0b1111)
#define FAST_MOD_32(x)  (x & 0b11111)
#define FAST_MOD_64(x)  (x & 0b111111)
#define FAST_MOD_128(x) (x & 0b1111111)
#define FAST_MOD_256(x) (x & 0b11111111)

// (x / 32)
#define FAST_DIV_2(x)   (x >> 1)
#define FAST_DIV_4(x)   (x >> 2)
#define FAST_DIV_8(x)   (x >> 3)
#define FAST_DIV_16(x)  (x >> 4)
#define FAST_DIV_32(x)  (x >> 5)
#define FAST_DIV_64(x)  (x >> 6)
#define FAST_DIV_128(x) (x >> 7)
#define FAST_DIV_256(x) (x >> 8)

// ceil(x/32)
#define CEIL_DIV_2(x)   ((x >> 1) + (  FAST_MOD_2(x) ? 1 : 0)) 
#define CEIL_DIV_4(x)   ((x >> 2) + (  FAST_MOD_4(x) ? 1 : 0)) 
#define CEIL_DIV_8(x)   ((x >> 3) + (  FAST_MOD_8(x) ? 1 : 0)) 
#define CEIL_DIV_16(x)  ((x >> 4) + ( FAST_MOD_16(x) ? 1 : 0)) 
#define CEIL_DIV_32(x)  ((x >> 5) + ( FAST_MOD_32(x) ? 1 : 0)) 
#define CEIL_DIV_64(x)  ((x >> 5) + ( FAST_MOD_64(x) ? 1 : 0)) 
#define CEIL_DIV_128(x) ((x >> 5) + (FAST_MOD_128(x) ? 1 : 0)) 
#define CEIL_DIV_256(x) ((x >> 5) + (FAST_MOD_256(x) ? 1 : 0)) 

// clamp(x, 32)
#define CLAMP_2(x)   (x & ~(0b1))
#define CLAMP_4(x)   (x & ~(0b11))
#define CLAMP_8(x)   (x & ~(0b111))
#define CLAMP_16(x)  (x & ~(0b1111))
#define CLAMP_32(x)  (x & ~(0b11111))
#define CLAMP_64(x)  (x & ~(0b111111))
#define CLAMP_128(x) (x & ~(0b1111111))
#define CLAMP_256(x) (x & ~(0b11111111))