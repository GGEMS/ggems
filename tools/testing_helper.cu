// GGEMS Copyright (C) 2017

/*!
 * \file testing.cuh
 * \brief Testing
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday October 10, 2016
 *
 */

#ifndef TESTING_HELPER_CU
#define TESTING_HELPER_CU


#include "testing_helper.cuh"

/////////////////////////////////////////////////////////////

__host__ __device__ void fun2_do_add( TestData data, CoefData coef, ui32 id )
{
    if (id==0) printf("InFun2 data size %i\n", data.n);
    data.C[id] = data.A[id] + data.B[id] * coef.coef[id];
}

__host__ __device__ void fun1_do_add( TestData data, CoefData coef, ui32 id )
{
    if (id==0) printf("InFun1 data size %i\n", data.n);
    fun2_do_add( data, coef, id );
}

__global__ void kernel_do_add_struct( TestData data, CoefData coef )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;;
    if( id >= data.n ) return;

    if (id==0) printf("InKernel data size %i\n", data.n);
    fun1_do_add( data, coef, id );
    if (id==0) printf("DoneKernel\n");
}


/////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////////////



#endif

















