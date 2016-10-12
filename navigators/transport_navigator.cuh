// GGEMS Copyright (C) 2015

/*!
 * \file transport_navigator.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 5 february 2016
 *
 *
 *
 */

#ifndef TRANSPORT_NAVIGATOR_CUH
#define TRANSPORT_NAVIGATOR_CUH

#include "particles.cuh"
#include "raytracing.cuh"
#include "primitives.cuh"
#include "voxelized.cuh"

// Get a safety position inside an AABB geometry
f32xyz __host__ __device__ transport_get_safety_inside_AABB(f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax,
                                                            f32 zmin, f32 zmax, f32 tolerance);

// Get a safety position outside an AABB geometry
f32xyz __host__ __device__ transport_get_safety_outside_AABB(f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax,
                                                             f32 zmin, f32 zmax, f32 tolerance);

// Compute safety position considering AABB geometry
__host__ __device__ f32 transport_compute_safety_AABB( f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax );

// Transport the current particle to an AABB geometry
__host__ __device__ void transport_track_to_in_AABB( ParticlesData &particles, f32 xmin, f32 xmax,
                                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 id );
__host__ __device__ void transport_track_to_in_AABB( ParticlesData &particles, const AabbData aabb, f32 tolerance, ui32 id );


#endif
