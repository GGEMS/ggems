// GGEMS Copyright (C) 2015

/*!
 * \file testing.cuh
 * \brief Testing
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday October 10, 2016
 *
 */

#ifndef TESTING_CU
#define TESTING_CU


#include "testing.cuh"

__global__ void kernel_do_add( f32* A, f32* B, f32* C, uint N )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;;
    if( id >= N ) return;

    C[id] = A[id]+B[id];
}

Testing::Testing()
{
    host_A = NULL;
    host_B = NULL;
    host_C = NULL;

    device_A = NULL;
    device_B = NULL;
    device_C = NULL;

    uni_A = NULL;
    uni_B = NULL;
    uni_C = NULL;

    N = 10;
}

Testing::~Testing()
{
//    free( host_A );
//    free( host_B );
//    free( host_C );

//    cudaFree( device_A );
//    cudaFree( device_B );
//    cudaFree( device_C );

//    cudaFree( uni_A );
//    cudaFree( uni_B );
//    cudaFree( uni_C );
}

void Testing::set_device_id(ui32 id)
{
    cudaSetDevice( id );
}

void Testing::info_device(i32 dev)
{
    i32 driverVersion = 0, runtimeVersion = 0;

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
}

void Testing::set_data_size(ui32 n)
{
    N = n;
}

void Testing::allocation_device()
{
    printf("Allocate one vector of %i Bytes on device: ", N*sizeof(f32));
    HANDLE_ERROR( cudaMalloc((void**) &device_A, N*sizeof(f32)) );
    printf("[ok]\n");

    printf("Free memory: ");
    cudaFree(device_A);
    printf("[ok]\n");
}

void Testing::copy_device()
{
    printf("Allocate vectors A and B (%i Bytes each on the host): ", N*sizeof(f32));
    host_A = (f32*)malloc(N*sizeof(f32));
    host_B = (f32*)malloc(N*sizeof(f32));
    printf("[ok]\n");

    printf("Set vector A to 1 and vector B to 0: ");
    ui32 i=0; while(i<N)
    {
        host_A[i] = 1.0;
        host_B[i] = 0.0;
        ++i;
    }
    printf("[ok]\n");

    printf("Host vector A: %f %f ... %f %f\n", host_A[0], host_A[1], host_A[N-2], host_A[N-1]);
    printf("Host vector B: %f %f ... %f %f\n", host_B[0], host_B[1], host_B[N-2], host_B[N-1]);

    printf("Allocate vector A of %i Bytes on device: ", N*sizeof(f32));
    HANDLE_ERROR( cudaMalloc((void**) &device_A, N*sizeof(f32)) );
    printf("[ok]\n");

    printf("Copy vector A (host) to vector A (device): ");
    HANDLE_ERROR( cudaMemcpy(device_A, host_A, N*sizeof(f32), cudaMemcpyHostToDevice) );
    printf("[ok]\n");

    printf("Copy back the vector A (device) to vector B (host): ");
    HANDLE_ERROR( cudaMemcpy(host_B, device_A, N*sizeof(f32), cudaMemcpyDeviceToHost) );
    printf("[ok]\n");

    printf("Results:\n");
    printf("Host vector A: %f %f ... %f %f\n", host_A[0], host_A[1], host_A[N-2], host_A[N-1]);
    printf("Host vector B: %f %f ... %f %f\n", host_B[0], host_B[1], host_B[N-2], host_B[N-1]);

    printf("Free memory: ");
    cudaFree(device_A);
    free(host_A);
    free(host_B);
    printf("[ok]\n");

}

void Testing::kernel_device()
{
    printf("Allocate vectors A, B and C (%i Bytes each on the host): ", N*sizeof(f32));
    host_A = (f32*)malloc(N*sizeof(f32));
    host_B = (f32*)malloc(N*sizeof(f32));
    host_C = (f32*)malloc(N*sizeof(f32));
    printf("[ok]\n");

    printf("Set vector A to 1, vector B to 2 and vector C to 0: ");
    ui32 i=0; while(i<N)
    {
        host_A[i] = 1.0;
        host_B[i] = 2.0;
        host_C[i] = 0.0;
        ++i;
    }
    printf("[ok]\n");

    printf("Allocate vectors A, B and C of %i Bytes on device: ", N*sizeof(f32));
    HANDLE_ERROR( cudaMalloc((void**) &device_A, N*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &device_B, N*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &device_C, N*sizeof(f32)) );
    printf("[ok]\n");

    printf("Copy vector A, B, C (host) to vector A, B, C (device): ");
    HANDLE_ERROR( cudaMemcpy(device_A, host_A, N*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_B, host_B, N*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_C, host_C, N*sizeof(f32), cudaMemcpyHostToDevice) );
    printf("[ok]\n");

    printf("Do C = A + B on device: ");
    dim3 threads, grid;
    threads.x = 128;
    grid.x = ( N + 128 - 1 ) / 128;

    kernel_do_add<<<grid, threads>>>( device_A, device_B, device_C, N );
    cuda_error_check( "Error ", " kernel_testing_do_add" );
    cudaDeviceSynchronize();


    printf("Copy back the vector C (device) to vector C (host): ");
    HANDLE_ERROR( cudaMemcpy(host_C, device_C, N*sizeof(f32), cudaMemcpyDeviceToHost) );
    printf("[ok]\n");

    printf("Results:\n");
    printf("Host vector A: %f %f ... %f %f\n", host_A[0], host_A[1], host_A[N-2], host_A[N-1]);
    printf("Host vector B: %f %f ... %f %f\n", host_B[0], host_B[1], host_B[N-2], host_B[N-1]);
    printf("Host vector C: %f %f ... %f %f\n", host_C[0], host_C[1], host_C[N-2], host_C[N-1]);

    printf("Free memory: ");
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(host_A);
    free(host_B);
    free(host_C);
    printf("[ok]\n");
}

void Testing::allocation_unified_memory()
{
    printf("Allocate one vector of %i Bytes on device: ", N*sizeof(f32));
    HANDLE_ERROR( cudaMallocManaged( &uni_A, N * sizeof( f32 ) ) );
    printf("[ok]\n");

    printf("Free memory: ");
    cudaFree(uni_A);
    printf("[ok]\n");
}

void Testing::kernel_unified_memory()
{
    printf("Allocate vectors A, B and C (%i Bytes each on the host): ", N*sizeof(f32));
    host_A = (f32*)malloc(N*sizeof(f32));
    host_B = (f32*)malloc(N*sizeof(f32));
    host_C = (f32*)malloc(N*sizeof(f32));
    printf("[ok]\n");

    printf("Allocate unified vectors A, B and C of %i Bytes: ", N*sizeof(f32));
    HANDLE_ERROR( cudaMallocManaged( &uni_A, N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &uni_B, N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &uni_C, N * sizeof( f32 ) ) );
    printf("[ok]\n");

    printf("Set vector A to 1, vector B to 2 and vector C to 0: ");
    ui32 i=0; while(i<N)
    {
        uni_A[i] = 1.0;
        uni_B[i] = 2.0;
        uni_C[i] = 0.0;
        ++i;
    }
    printf("[ok]\n");

    printf("Do C = A + B on device: ");
    dim3 threads, grid;
    threads.x = 128;
    grid.x = ( N + 128 - 1 ) / 128;

    kernel_do_add<<<grid, threads>>>( uni_A, uni_B, uni_C, N );
    cuda_error_check( "Error ", " kernel_testing_do_add" );
    cudaDeviceSynchronize();

    printf("Results:\n");
    printf("vector A: %f %f ... %f %f\n", uni_A[0], uni_A[1], uni_A[N-2], uni_A[N-1]);
    printf("vector B: %f %f ... %f %f\n", uni_B[0], uni_B[1], uni_B[N-2], uni_B[N-1]);
    printf("vector C: %f %f ... %f %f\n", uni_C[0], uni_C[1], uni_C[N-2], uni_C[N-1]);

    printf("Free memory: ");
    cudaFree(uni_A);
    cudaFree(uni_B);
    cudaFree(uni_C);
    printf("[ok]\n");
}

#endif

















