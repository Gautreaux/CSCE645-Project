// Test endianness of both the CUDA device and the local device

#include <iostream>

#include <cstdint>

// packs output with 0 if big-endian
//  or -1u if little endian
__host__ __device__ void checkEndian(char* output){
    *output = 0;

    int32_t a = 0x00010203;
    if(*((char*)(&a)) == 0x03){
        // little endian
        *output = ~(*output);
    }
}


__global__ void checkEndianEntry(char* output){
    checkEndian(output);
}


int main(void){
    std::cout << "Hello world\n" << std::endl;

    char host_endian;
    checkEndian(&host_endian);

    if(host_endian == 0){
        std::cout << "Host is big-endian" << std::endl;
    }
    else{
        std::cout << "Host is little-endian" << std::endl;
    }

    // now check the device
    char device_endian;
    char* device_buffer;
    cudaMalloc(&device_buffer, sizeof(char));
    checkEndianEntry<<<1,1>>>(device_buffer);
    cudaMemcpy(&device_endian, device_buffer, sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(device_buffer);

    if(device_endian == 0){
        std::cout << "Device is big-endian" << std::endl;
    }
    else{
        std::cout << "Device is little-endian" << std::endl;
    }
}