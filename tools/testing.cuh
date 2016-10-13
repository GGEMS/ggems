// GGEMS Copyright (C) 2015

/*!
 * \file testing.cuh
 * \brief testing
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday October 10, 2016
 *
 */

#ifndef TESTING_CUH
#define TESTING_CUH

#include "global.cuh"

// Cross section table for photon particle
struct TestData{
    f32 *A;
    f32 *B;
    f32 *C;
    ui32 n;
};

// Struct that handle CPU&GPU CS data
struct StructTest {
    TestData data_h;
    TestData data_d;
};

class Testing
{
public:   
    Testing();
    ~Testing();

    void set_device_id( ui32 id );
    void set_data_size( ui32 n );
    void info_device(i32 dev);

    void allocation_device();
    void kernel_device();
    void copy_device();

    void allocation_unified_memory();
    void kernel_unified_memory();

    void kernel_struct();

private:
    f32 *host_A;
    f32 *host_B;
    f32 *host_C;

    f32 *device_A;
    f32 *device_B;
    f32 *device_C;

    f32 *uni_A;
    f32 *uni_B;
    f32 *uni_C;

    ui32 N;

    StructTest st_test;
};

#endif
