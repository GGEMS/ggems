#ifndef GUARD_GGEMS_TOOLS_GGTYPES_HH
#define GUARD_GGEMS_TOOLS_GGTYPES_HH

/*!
  \file GGTypes.hh

  \brief Redefining types for OpenCL device and host

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#ifdef OPENCL_COMPILER // On OpenCL device
#define GGchar char
#define GGuchar unsigned char
#define GGushort short
#define GGushort unsigned short
#define GGint int
#define GGuint unsigned int
#define GGlong long
#define GGulong unsigned long

#define GGfloat float
#define GGfloat2 float2
#define GGfloat3 float3
#define GGfloat4 float4
#define GGfloat8 float8
#define GGfloat16 float16

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define GGdouble double
#define GGdouble2 double2
#define GGdouble3 double3
#define GGdouble4 double4
#define GGdouble8 double8
#define GGdouble16 double16
#else // On host device
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define GGbool bool
#define GGchar cl_char
#define GGuchar cl_uchar
#define GGshort cl_short
#define GGushort cl_ushort
#define GGint cl_int
#define GGuint cl_uint
#define GGlong cl_long
#define GGulong cl_ulong

#define GGfloat cl_float
#define GGfloat2 cl_float2
#define GGfloat3 cl_float3
#define GGfloat4 cl_float4
#define GGfloat8 cl_float8
#define GGfloat16 cl_float16

#define GGdouble cl_double
#define GGdouble2 cl_double2
#define GGdouble3 cl_double3
#define GGdouble4 cl_double4
#define GGdouble8 cl_double8
#define GGdouble16 cl_double16
#endif

#endif // End of GUARD_GGEMS_OPENCL_GGTYPES_HH
