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

// Transport the current particle to an AABB geometry
__host__ __device__ void transport_track_to_in_AABB( ParticlesData &particles, f32 xmin, f32 xmax,
                                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 id );


#endif
