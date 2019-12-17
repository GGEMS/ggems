#ifndef GUARD_GGEMS_OPENCL_TYPES_HH
#define GUARD_GGEMS_OPENCL_TYPES_HH

/*!
  \file types.hh

  \brief Redefining types for OpenCL device and host

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

// Redefine type for instance cl_float compiling on host and float compiling
// on OpenCL device
#ifdef OPENCL_COMPILER

#define f64cl_t double
#define f32cl_t float
#define uintcl_t unsigned int
#define ushortcl_t unsigned short
#define ucharcl_t unsigned char

#define f323cl_t float3

#else

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define f64cl_t cl_double
#define f32cl_t cl_float
#define uintcl_t cl_uint
#define ushortcl_t cl_ushort
#define ucharcl_t cl_uchar

#define f323cl_t cl_float3

#endif

#endif // End of GUARD_GGEMS_OPENCL_TYPES_HH
