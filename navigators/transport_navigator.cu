// GGEMS Copyright (C) 2017

/*!
 * \file transport_navigator.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 5 february 2016
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 */

#ifndef TRANSPORT_NAVIGATOR_CU
#define TRANSPORT_NAVIGATOR_CU

#include "transport_navigator.cuh"

// Get a safety position inside an AABB geometry
f32xyz __host__ __device__ transport_get_safety_inside_AABB( f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax,
                                                             f32 zmin, f32 zmax, f32 tolerance )
{
    // on x
    f32 SafXmin = fabs( pos.x-xmin );
    f32 SafXmax = fabs( pos.x-xmax );

    pos.x = ( SafXmin < tolerance ) ?  xmin+tolerance : pos.x;
    pos.x = ( SafXmax < tolerance ) ?  xmax-tolerance : pos.x;

    // on y
    f32 SafYmin = fabs( pos.y-ymin );
    f32 SafYmax = fabs( pos.y-ymax );

    pos.y = ( SafYmin < tolerance ) ?  ymin+tolerance : pos.y;
    pos.y = ( SafYmax < tolerance ) ?  ymax-tolerance : pos.y;

    // on z
    f32 SafZmin = fabs( pos.z-zmin );
    f32 SafZmax = fabs( pos.z-zmax );

    pos.z = ( SafZmin < tolerance ) ?  zmin+tolerance : pos.z;
    pos.z = ( SafZmax < tolerance ) ?  zmax-tolerance : pos.z;

    return pos;
}

// Get a safety position outside an AABB geometry
f32xyz __host__ __device__ transport_get_safety_outside_AABB( f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax,
                                                              f32 zmin, f32 zmax, f32 tolerance )
{
    // on x
    f32 SafXmin = fabs( pos.x-xmin );
    f32 SafXmax = fabs( pos.x-xmax );

    pos.x = ( SafXmin < tolerance ) ?  xmin-tolerance : pos.x;
    pos.x = ( SafXmax < tolerance ) ?  xmax+tolerance : pos.x;

    // on y
    f32 SafYmin = fabs( pos.y-ymin );
    f32 SafYmax = fabs( pos.y-ymax );

    pos.y = ( SafYmin < tolerance ) ?  ymin-tolerance : pos.y;
    pos.y = ( SafYmax < tolerance ) ?  ymax+tolerance : pos.y;

    // on z
    f32 SafZmin = fabs( pos.z-zmin );
    f32 SafZmax = fabs( pos.z-zmax );

    pos.z = ( SafZmin < tolerance ) ?  zmin-tolerance : pos.z;
    pos.z = ( SafZmax < tolerance ) ?  zmax+tolerance : pos.z;

    return pos;
}

// Get safety position considering AABB geometry
f32 __host__ __device__ transport_compute_safety_AABB( f32xyz pos, f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{
    f32 safety;
    safety = fminf( fabs( pos.x-xmin ), fabs( pos.x-xmax ) );
    safety = fminf( safety, fminf( fabs( pos.y-ymin ), fabs( pos.y-ymax ) ) );
    safety = fminf( safety, fminf( fabs( pos.z-zmin ), fabs( pos.z-zmax ) ) );
    return safety;
}

// Transport the current particle to an AABB geometry
void __host__ __device__ transport_track_to_in_AABB( ParticlesData *particles, f32 xmin, f32 xmax,
                                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 id)
{

    // If freeze (not dead), re-activate the current particle
    if( particles->status[ id ] == PARTICLE_FREEZE )
    {
        particles->status[ id ] = PARTICLE_ALIVE;
    }
    else if ( particles->status[ id ] == PARTICLE_DEAD )
    {
        return;
    }

    // Read position
    f32xyz pos;
    pos.x = particles->px[ id ];
    pos.y = particles->py[ id ];
    pos.z = particles->pz[ id ];

    // Read direction
    f32xyz dir;
    dir.x = particles->dx[ id ];
    dir.y = particles->dy[ id ];
    dir.z = particles->dz[ id ];

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in pos %f %f %f\n", id, pos.x, pos.y, pos.z );
        }
#endif

    // Skip if already inside the phantom
    if ( test_point_AABB_with_tolerance (pos, xmin, xmax, ymin, ymax, zmin, zmax, tolerance ) )
    {

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in already inside the phantom\n", id );
        }
#endif

        return;
    }

    // get distance to AABB
    f32 dist = hit_ray_AABB ( pos, dir, xmin, xmax, ymin, ymax, zmin, zmax );    

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in dist to AABB %f\n", id, dist );
        }
#endif

    // the particle not hitting the voxelized volume
    if ( dist == FLT_MAX )                            // TODO: Don't know why F32_MAX doesn't work...
    {
        particles->status[ id ] = PARTICLE_FREEZE;

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in not hitting: FREEZE\n", id );
        }
#endif

        return;
    }
    else
    {
        // Check if the path of the particle cross the volume sufficiently
        f32 cross = dist_overlap_ray_AABB ( pos, dir, xmin, xmax, ymin, ymax, zmin, zmax );

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in crossing value: %f\n", id, cross );
        }
#endif


        if ( cross < ( 2*tolerance ) )
        {
            particles->status[ id ] = PARTICLE_FREEZE;

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in crossing to small: FREEZE\n", id );
        }
#endif
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add ( pos, fxyz_scale ( dir, dist+tolerance ) );

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in move to pos %f %f %f\n", id, pos.x, pos.y, pos.z );
        }
#endif

        // correct the particle position if not totally inside due to float tolerance
        pos = transport_get_safety_inside_AABB( pos, xmin, xmax, ymin, ymax, zmin, zmax, tolerance );

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in safety to pos %f %f %f\n", id, pos.x, pos.y, pos.z );
        }
#endif

        // update tof
        particles->tof[ id ] += c_light * dist;

        // set the new position
        particles->px[ id ] = pos.x;
        particles->py[ id ] = pos.y;
        particles->pz[ id ] = pos.z;

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in values assigned\n", id );
        }
#endif

        // reset geom id
        particles->geometry_id[ id ] = 0;

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("  ID %i in track2in geom index to 0\n", id );
        }
#endif


    }

}

void __host__ __device__ transport_track_to_in_AABB( ParticlesData *particles, AabbData aabb, f32 tolerance, ui32 id)
{
    transport_track_to_in_AABB( particles, aabb.xmin, aabb.xmax, aabb.ymin, aabb.ymax, aabb.zmin, aabb.zmax, tolerance, id);
}

/*
// Transport the current particle to an OBB geometry
void __host__ __device__ transport_track_to_in_OBB( ParticlesData &particles, ObbData obb, ui32 id)
                                                    //f32 xmin, f32 xmax,
                                                    // f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 id)
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

    // Transform the ray in OBB' space, then do AABB
    f32xyz tmp_pos = fxyz_sub(pos, obb.center);
    pos.x = fxyz_dot(tmp_pos, obb.u);
    pos.y = fxyz_dot(tmp_pos, obb.v);
    pos.z = fxyz_dot(tmp_pos, obb.w);
    f32xyz tmp_dir = dir;
    dir.x = fxyz_dot(tmp_dir, obb.u);
    dir.y = fxyz_dot(tmp_dir, obb.v);
    dir.z = fxyz_dot(tmp_dir, obb.w);

    // get distance to AABB
    f32 dist = hit_ray_AABB ( pos, dir, obb.xmin, obb.xmax, obb.ymin, obb.ymax, obb.zmin, obb.zmax );

    // the particle not hitting the voxelized volume
    if ( dist == FLT_MAX )                            // TODO: Don't know why F32_MAX doesn't work...
    {
        particles.endsimu[id] = PARTICLE_FREEZE;
        return;
    }
    else
    {
        // Check if the path of the particle cross the volume sufficiently
        f32 cross = dist_overlap_ray_AABB ( pos, dir, obb.xmin, obb.xmax, obb.ymin, obb.ymax, obb.zmin, obb.zmax );
        if ( cross < ( 2*TOLERANCE ) )
        {
            particles.endsimu[id] = PARTICLE_FREEZE;
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add ( pos, fxyz_scale ( dir, dist+TOLERANCE ) );

        // correct the particle position if not totally inside due to float tolerance
        pos = transport_get_safety_inside_AABB( pos, obb.xmin, obb.xmax, obb.ymin, obb.ymax, obb.zmin, obb.zmax );

        // update tof
        particles.tof[id] += c_light * dist;

    }

    // Transform back the ray to world space
    f32xyz u, v, w;
    u.x = 1.0f; u.y = 0.0f; u.z = 0.0f;
    v.x = 0.0f; v.y = 1.0f; v.z = 0.0f;
    w.x = 0.0f; w.y = 0.0f; w.z = 1.0f;

    f32xyz new_pos;
    new_pos.x = fxyz_dot(pos, u);
    new_pos.y = fxyz_dot(pos, v);
    new_pos.z = fxyz_dot(pos, w);

    new_pos = fxyz_add(new_pos, obb.center);

    // set the new position
    particles.px[id] = new_pos.x;
    particles.py[id] = new_pos.y;
    particles.pz[id] = new_pos.z;

}
*/


/*
void __host__ __device__ transport_move_inside_AABB( ParticlesData &particles, f32xyz pos, f32xyz dir,
                                                     f32 next_interaction_distance,
                                                     f32 xmin, f32 xmax,
                                                     f32 ymin, f32 ymax,
                                                     f32 zmin, f32 zmax, ui32 id )
{
    // move to the next position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // get safety position


    // update tof
    particles.tof[id] += c_light * next_interaction_distance;

    // store new pos
    particles.px[id] = pos.x;
    particles.py[id] = pos.y;
    particles.pz[id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance (pos, xmin, xmax, ymin, ymax, zmin, zmax, TOLERANCE ) )
    {
        particles.endsimu[id] = PARTICLE_FREEZE;
    }

}
*/

#endif
