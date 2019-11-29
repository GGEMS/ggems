#ifndef GUARD_USE_DOUBLE_PRECISION_HH
#define GUARD_USE_DOUBLE_PRECISION_HH

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#endif // GUARD_USE_DOUBLE_PRECISION_HH
