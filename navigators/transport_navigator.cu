// GGEMS Copyright (C) 2015

/*!
 * \file transport_navigator.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 5 february 2016
 *
 *
 *
 */

#ifndef TRANSPORT_NAVIGATOR_CU
#define TRANSPORT_NAVIGATOR_CU

#include "transport_navigator.cuh"

// Transport the current particle to an AABB geometry
void __host__ __device__ transport_track_to_in_AABB( ParticlesData &particles, f32 xmin, f32 xmax,
                                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 id)
{

    // Read position
    f64xyz pos;
    pos.x = particles.px[id];
    pos.y = particles.py[id];
    pos.z = particles.pz[id];

    // Read direction
    f64xyz dir;
    dir.x = particles.dx[id];
    dir.y = particles.dy[id];
    dir.z = particles.dz[id];

    f32 dist = hit_ray_AABB ( pos, dir, xmin, xmax, ymin, ymax, zmin, zmax );

    // the particle not hitting the voxelized volume
    if ( dist == FLT_MAX )                            // TODO: Don't know why F32_MAX doesn't work...
    {
        particles.endsimu[id] = PARTICLE_FREEZE;
        return;
    }
    else
    {
        // Check if the path of the particle cross the volume sufficiently
        f32 cross = dist_overlap_ray_AABB ( pos, dir, xmin, xmax, ymin, ymax, zmin, zmax );
        if ( cross < EPSILON3 )
        {
            particles.endsimu[id] = PARTICLE_FREEZE;
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add ( pos, fxyz_scale ( dir, dist+EPSILON3 ) );

        // update tof
        particles.tof[id] += c_light * dist;

        // set the new position
        particles.px[id] = pos.x;
        particles.py[id] = pos.y;
        particles.pz[id] = pos.z;
    }

}

#endif
