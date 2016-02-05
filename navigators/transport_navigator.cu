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


// Get safety position considering AABB geometry
f32xyz __host__ __device__ transport_get_safety_AABB( f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{
    // on x
    f32 SafXmin = fabs( pos.x-xmin );
    f32 SafXmax = fabs( pos.x-xmax );

    pos.x = ( SafXmin < EPSILON3 ) ?  xmin+EPSILON3 : pos.x;
    pos.x = ( SafXmax < EPSILON3 ) ?  xmax-EPSILON3 : pos.x;

    // on y
    f32 SafYmin = fabs( pos.y-ymin );
    f32 SafYmax = fabs( pos.y-ymax );

    pos.y = ( SafYmin < EPSILON3 ) ?  ymin+EPSILON3 : pos.y;
    pos.y = ( SafYmax < EPSILON3 ) ?  ymax-EPSILON3 : pos.y;

    // on z
    f32 SafZmin = fabs( pos.z-zmin );
    f32 SafZmax = fabs( pos.z-zmax );

    pos.z = ( SafZmin < EPSILON3 ) ?  zmin+EPSILON3 : pos.z;
    pos.z = ( SafZmax < EPSILON3 ) ?  zmax-EPSILON3 : pos.z;

    return pos;
}

// Get safety position considering AABB geometry
f32xyz __host__ __device__ transport_compute_safety_AABB( f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{
    f32xyz safety;
    safety.x = fmin( fabs( pos.x-xmin ), fabs( pos.x-xmax ) );
    safety.y = fmin( fabs( pos.y-ymin ), fabs( pos.y-ymax ) );
    safety.z = fmin( fabs( pos.z-zmin ), fabs( pos.z-zmax ) );
    return safety;
}

// Transport the current particle to an AABB geometry
void __host__ __device__ transport_track_to_in_AABB( ParticlesData &particles, f32 xmin, f32 xmax,
                                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 id)
{

    // Read position
    f32xyz pos;
    pos.x = particles.px[id];
    pos.y = particles.py[id];
    pos.z = particles.pz[id];

    // Read direction
    f32xyz dir;
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
        if ( cross < ( 2*EPSILON3 ) )
        {
            particles.endsimu[id] = PARTICLE_FREEZE;
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add ( pos, fxyz_scale ( dir, dist+EPSILON3 ) );

        // correct the particle position if not totally inside due to float tolerance
        pos = transport_get_safety_AABB( pos, xmin, xmax, ymin, ymax, zmin, zmax );

        // update tof
        particles.tof[id] += c_light * dist;

        // set the new position
        particles.px[id] = pos.x;
        particles.py[id] = pos.y;
        particles.pz[id] = pos.z;
    }

}

#endif
