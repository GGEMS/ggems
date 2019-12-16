#ifndef GUARD_GGEMS_OPENCL_USE_DOUBLE_PRECISION_HH
#define GUARD_GGEMS_OPENCL_USE_DOUBLE_PRECISION_HH

#ifdef OPENCL_COMPILER
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
#endif

#endif // End of GUARD_GGEMS_OPENCL_USE_DOUBLE_PRECISION_HH
