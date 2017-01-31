// GGEMS Copyright (C) 2017

/*!
 * \file testing.cuh
 * \brief testing
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday October 10, 2016
 *
 */

#ifndef TESTING_HELPER_CUH
#define TESTING_HELPER_CUH

#include "global.cuh"

struct TestData{
    f32 *A;
    f32 *B;
    f32 *C;
    ui32 n;
};

struct StructTest {
    TestData data_h;
    TestData data_d;
};

struct CoefData
{
    f32* coef;
    ui32 n;
};

struct StructCoef {
    CoefData data_h;
    CoefData data_d;
};


__host__ __device__ void fun2_do_add( TestData data, CoefData coef, ui32 id );
__host__ __device__ void fun1_do_add( TestData data, CoefData coef, ui32 id );
__global__ void kernel_do_add_struct( TestData data, CoefData coef );

//class TestingHelper
//{
//public:
//    TestingHelper() {}
//    ~TestingHelper() {}

//    void launch_kernel_do_add_struct( StructTest st_test, StructCoef st_coef );

//private:

//};

#endif
