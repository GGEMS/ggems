#ifndef GUARD_GGEMS_TOOLS_GGEMSTYPES_HH
#define GUARD_GGEMS_TOOLS_GGEMSTYPES_HH

/*!
  \file GGEMSTypes.hh

  \brief Redefining types for OpenCL device and host

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#ifdef OPENCL_COMPILER // On OpenCL device
#define GGchar char
#define GGchar2 char2
#define GGchar3 char3
#define GGchar4 char4
#define GGchar8 char8
#define GGchar16 char16

#define GGuchar uchar
#define GGuchar2 uchar2
#define GGuchar3 uchar3
#define GGuchar4 uchar4
#define GGuchar8 uchar8
#define GGuchar16 uchar16

#define GGushort short
#define GGshort2 short2
#define GGshort3 short3
#define GGshort4 short4
#define GGshort8 short8
#define GGshort16 short16

#define GGushort ushort
#define GGushort2 ushort2
#define GGushort3 ushort3
#define GGushort4 ushort4
#define GGushort8 ushort8
#define GGushort16 ushort16

#define GGint int
#define GGint2 int2
#define GGint3 int3
#define GGint4 int4
#define GGint8 int8
#define GGint16 int16

#define GGuint uint
#define GGuint2 uint2
#define GGuint3 uint3
#define GGuint4 uint4
#define GGuint8 uint8
#define GGuint16 uint16

#define GGlong long
#define GGlong2 long2
#define GGlong3 long3
#define GGlong4 long4
#define GGlong8 long8
#define GGlong16 long16

#define GGulong ulong
#define GGulong2 ulong2
#define GGulong3 ulong3
#define GGulong4 ulong4
#define GGulong8 ulong8
#define GGulong16 ulong16

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
#define GGchar2 cl_char2
#define GGchar3 cl_char3
#define GGchar4 cl_char4
#define GGchar8 cl_char8
#define GGchar16 cl_char16

#define GGuchar cl_uchar
#define GGuchar2 cl_uchar2
#define GGuchar3 cl_uchar3
#define GGuchar4 cl_uchar4
#define GGuchar8 cl_uchar8
#define GGuchar16 cl_uchar16

#define GGshort cl_short
#define GGshort2 cl_short2
#define GGshort3 cl_short3
#define GGshort4 cl_short4
#define GGshort8 cl_short8
#define GGshort16 cl_short16

#define GGushort cl_ushort
#define GGushort2 cl_ushort2
#define GGushort3 cl_ushort3
#define GGushort4 cl_ushort4
#define GGushort8 cl_ushort8
#define GGushort16 cl_ushort16

#define GGint cl_int
#define GGint2 cl_int2
#define GGint3 cl_int3
#define GGint4 cl_int4
#define GGint8 cl_int8
#define GGint16 cl_int16

#define GGuint cl_uint
#define GGuint2 cl_uint2
#define GGuint3 cl_uint3
#define GGuint4 cl_uint4
#define GGuint8 cl_uint8
#define GGuint16 cl_uint16

#define GGlong cl_long
#define GGlong2 cl_long2
#define GGlong3 cl_long3
#define GGlong4 cl_long4
#define GGlong8 cl_long8
#define GGlong16 cl_long16

#define GGulong cl_ulong
#define GGulong2 cl_ulong2
#define GGulong3 cl_ulong3
#define GGulong4 cl_ulong4
#define GGulong8 cl_ulong8
#define GGulong16 cl_ulong16

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

/*!
  \fn inline GGfloat MakeFloat3(GGfloat const x, GGfloat const y, GGfloat const z)
  \param x - x parameter
  \param y - y parameter
  \param z - z parameter
  \brief Make a float X, Y and Z with custom values
*/
inline GGfloat3 MakeFloat3(GGfloat const x, GGfloat const y, GGfloat const z)
{
  GGfloat3 tmp;
  #ifdef OPENCL_COMPILER
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  #else
  tmp.s[0] = x;
  tmp.s[1] = y;
  tmp.s[2] = z;
  #endif
  return tmp;
}

/*!
  \fn inline GGfloat3 MakeFloat3Zeros()
  \brief Make a float X, Y and Z with zeros for value
*/
inline GGfloat3 MakeFloat3Zeros()
{
  GGfloat3 tmp;
  #ifdef OPENCL_COMPILER
  tmp.x = 0.0f;
  tmp.y = 0.0f;
  tmp.z = 0.0f;
  #else
  tmp.s[0] = 0.0f;
  tmp.s[1] = 0.0f;
  tmp.s[2] = 0.0f;
  #endif
  return tmp;
}

#endif // End of GUARD_GGEMS_TOOLS_GGEMSTYPES_HH
