// GGEMS Copyright (C) 2015

/*!
 * \file mesh_phan_linac_nav.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday June 13, 2016
 *
 * v0.1: JB - First code
 *
 */

#ifndef MESH_PHAN_LINAC_NAV_CU
#define MESH_PHAN_LINAC_NAV_CU

#include "mesh_phan_linac_nav.cuh"

////// HOST-DEVICE GPU Codes //////////////////////////////////////////////////////////////

// == Track to in ===================================================================================

// Device Kernel that move particles to the voxelized volume boundary
__global__ void MPLINACN::kernel_device_track_to_in( ParticlesData particles, LinacData linac, f32 geom_tolerance )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    // read position and direction
    f32xyz pos = make_f32xyz( particles.px[ id ], particles.py[ id ], particles.pz[ id ] );
    f32xyz dir = make_f32xyz( particles.dx[ id ], particles.dy[ id ], particles.dz[ id ] );

    // Change the frame to the particle (global to linac)
    pos = fxyz_global_to_local_position( linac.transform, pos );
    dir = fxyz_global_to_local_direction( linac.transform, dir );

    // Store data
    particles.px[ id ] = pos.x;
    particles.py[ id ] = pos.y;
    particles.pz[ id ] = pos.z;
    particles.dx[ id ] = dir.x;
    particles.dy[ id ] = dir.y;
    particles.dz[ id ] = dir.z;

    transport_track_to_in_AABB( particles, linac.aabb, geom_tolerance, id );

    // Start outside a mesh
    particles.geometry_id[ id ] = 0;  // first Byte set to zeros (outside a mesh)
}

// Host Kernel that move particles to the voxelized volume boundary
void MPLINACN::kernel_host_track_to_in( ParticlesData particles, LinacData linac, f32 geom_tolerance, ui32 id )
{
    // read position and direction
    f32xyz pos = make_f32xyz( particles.px[ id ], particles.py[ id ], particles.pz[ id ] );
    f32xyz dir = make_f32xyz( particles.dx[ id ], particles.dy[ id ], particles.dz[ id ] );

//    printf("%i Track2in: pos %f %f %f\n", id, pos.x, pos.y, pos.z);

    // Change the frame to the particle (global to linac)
    pos = fxyz_global_to_local_position( linac.transform, pos );
    dir = fxyz_global_to_local_direction( linac.transform, dir );

    // Store data
    particles.px[ id ] = pos.x;
    particles.py[ id ] = pos.y;
    particles.pz[ id ] = pos.z;
    particles.dx[ id ] = dir.x;
    particles.dy[ id ] = dir.y;
    particles.dz[ id ] = dir.z;

    transport_track_to_in_AABB( particles, linac.aabb, geom_tolerance, id );

    // Start outside a mesh
    particles.geometry_id[ id ] = 0;  // first Byte set to zeros (outside a mesh)

//    printf("%i transport: pos %f %f %f  status %i\n", id, pos.x, pos.y, pos.z, particles.endsimu[ id ]);
}

// == Track to out ===================================================================================

//            32   28   24   20   16   12   8    4
// geometry:  0000 0000 0000 0000 0000 0000 0000 0000
//            \__/ \____________/ \_________________/
//             |         |                 |
//            nav mesh  type of geometry  geometry index

__host__ __device__ ui16 m_read_geom_type( ui32 geometry )
{
    return ui16( ( geometry & 0x0FFF0000 ) >> 16 );
}

__host__ __device__ ui16 m_read_geom_index( ui32 geometry )
{
    return ui16( geometry & 0x0000FFFF );
}

__host__ __device__ ui8 m_read_geom_nav( ui32 geometry )
{
    return ui8( ( geometry & 0xF0000000 ) >> 28 );
}

__host__ __device__ ui32 m_write_geom_type( ui32 geometry, ui16 type )
{
    //              mask           write   shift
    return ( geometry & 0xF000FFFF ) | ( type << 16 ) ;
}

__host__ __device__ ui32 m_write_geom_index( ui32 geometry, ui16 index )
{
    //              mask           write   shift
    return ( geometry & 0xFFFF0000 ) | index ;
}

__host__ __device__ ui32 m_write_geom_nav( ui32 geometry, ui8 nav )
{
    //              mask           write   shift
    return ( geometry & 0x0FFFFFFF ) | ( nav << 28 ) ;
}

__host__ __device__ void m_transport_mesh( f32xyz pos, f32xyz dir,
                                           f32xyz *v1, f32xyz *v2, f32xyz *v3, ui32 offset, ui32 nb_tri,
                                           f32 geom_tol,
                                           bool *inside, bool *hit, f32 *distance )
{
    f32 cur_distance, tmin, tmax;

    tmin =  FLT_MAX;
    tmax = -FLT_MAX;

    // Loop over triangles
    ui32 itri = 0; while ( itri < nb_tri )
    {
        cur_distance = hit_ray_triangle( pos, dir, v1[ offset+itri ], v2[ offset+itri ], v3[ offset+itri ] );
        tmin = ( cur_distance < tmin ) ? cur_distance : tmin;
        tmax = ( cur_distance > tmax ) ? cur_distance : tmax;
        ++itri;
    }

    // Analyse tmin and tmax

    //   tmin = tmax = 0
    // -------(+)------>
    //       (   )
    if ( tmin < 0.0 && tmin > -geom_tol && tmax > 0.0 && tmax < geom_tol  )
    {
        *inside = false;
        *hit = false;
        *distance = FLT_MAX;
        return;
    }

    //
    //  tmin +inf   tmax -inf
    //
    // ---+----->
    //
    //  (     )
    if ( tmin > 0.0 && tmax < 0.0 )
    {
        *inside = false;
        *hit = false;
        *distance = FLT_MAX;
        return;
    }

    //    tmin       tmax
    //  ----(----+----)--->
    if ( tmin < 0.0 && tmax > 0.0 )
    {
        *inside = true;
        *hit = true;
        *distance = tmax;
        return;
    }

    //      tmin   tmax
    // --+---(------)--->
    if ( tmin > 0.0 && tmax > 0.0 )
    {
        *inside = false;
        *hit = true;
        *distance = tmin;
        return;
    }

    //     tmin   tmax
    // -----(-------)--+--->
    if ( tmin < 0.0 && tmax < 0.0 )
    {
        *inside = false;
        *hit = false;
        *distance = FLT_MAX;
        return;
    }

}

__host__ __device__ void m_mlc_nav_out_mesh( f32xyz pos, f32xyz dir, LinacData linac,
                                             f32 geom_tol,
                                             ui32 *geometry_id, f32 *geometry_distance )
{



    // First check where is the particle /////////////////////////////////////////////////////

    ui16 in_obj = IN_NOTHING;

    if ( linac.X_nb_jaw != 0 )
    {
        if ( test_point_AABB( pos, linac.X_jaw_aabb[ 0 ] ) )
        {
            in_obj = IN_JAW_X1;
        }
        if ( test_point_AABB( pos, linac.X_jaw_aabb[ 1 ] ) )
        {
            in_obj = IN_JAW_X2;
        }
    }

    if ( linac.Y_nb_jaw != 0 )
    {
        if ( test_point_AABB( pos, linac.Y_jaw_aabb[ 0 ] ) )
        {
            in_obj = IN_JAW_Y1;
        }
        if ( test_point_AABB( pos, linac.Y_jaw_aabb[ 1 ] ) )
        {
            in_obj = IN_JAW_Y2;
        }
    }

    if ( test_point_AABB( pos, linac.A_bank_aabb ) )
    {
        in_obj = IN_BANK_A;
    }

    if ( test_point_AABB( pos, linac.B_bank_aabb ) )
    {
        in_obj = IN_BANK_B;
    }

//    printf( " ---# In aabb %i\n", in_obj );

    // If the particle is outside the MLC element, then get the clostest bounding box //////////

    *geometry_distance = FLT_MAX;
    //*geometry_id = 0;
    //ui8 navigation;

    f32 distance = FLT_MAX;

    if ( in_obj == IN_NOTHING )
    {
        // Mother volume (AABB of the LINAC)
        *geometry_distance = hit_ray_AABB( pos, dir, linac.aabb );

//        if (*geometry_distance < 0.0) printf(" LINAC WARNING %f\n", *geometry_distance);

//        printf("  dist to linac aabb %f\n", *geometry_distance );

        if ( linac.X_nb_jaw != 0 )
        {
            distance = hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 0 ] );
            if ( distance < *geometry_distance )
            {
                *geometry_distance = distance;
            }

//            if (*geometry_distance < 0.0) printf(" JAW X1 WARNING %f\n", *geometry_distance);

            distance = hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 1 ] );
            if ( distance < *geometry_distance )
            {
                *geometry_distance = distance;
            }
        }

        if ( linac.Y_nb_jaw != 0 )
        {
            distance = hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 0 ] );
            if ( distance < *geometry_distance )
            {
                *geometry_distance = distance;
            }

            distance = hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 1 ] );
            if ( distance < *geometry_distance )
            {
                *geometry_distance = distance;
            }
        }

        distance = hit_ray_AABB( pos, dir, linac.A_bank_aabb );
        if ( distance < *geometry_distance )
        {
            *geometry_distance = distance;
        }

        distance = hit_ray_AABB( pos, dir, linac.B_bank_aabb );
        if ( distance < *geometry_distance )
        {
            *geometry_distance = distance;
        }

        // Store data and return
        *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
        *geometry_id = m_write_geom_type( *geometry_id, in_obj );

        return;
    }

    // Else, particle within a bounding box, need to get the closest distance to the mesh

    else
    {
        ui32 ileaf;
        bool inside_mesh = false;
        bool hit_mesh = false;
        i16 geom_index = -1;
        *geometry_distance = FLT_MAX;

        if ( in_obj == IN_JAW_X1 )
        {
            m_transport_mesh( pos, dir, linac.X_jaw_v1, linac.X_jaw_v2, linac.X_jaw_v3,
                              linac.X_jaw_index[ 0 ], linac.X_jaw_nb_triangles[ 0 ], geom_tol,
                              &inside_mesh, &hit_mesh, &distance );

            // If already inside the mesh
            if ( inside_mesh )
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_X1 );
                *geometry_distance = 0.0;
                return;
            }
            else if ( hit_mesh ) // Outside and hit the mesh
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_X1 );
                *geometry_distance = distance;
                return;
            }
            else // Not inside not hitting (then get the AABB distance)
            {
                *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_NOTHING );
                *geometry_distance = hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 0 ] );
                return;
            }
        }

        if ( in_obj == IN_JAW_X2 )
        {
            m_transport_mesh( pos, dir, linac.X_jaw_v1, linac.X_jaw_v2, linac.X_jaw_v3,
                              linac.X_jaw_index[ 1 ], linac.X_jaw_nb_triangles[ 1 ], geom_tol,
                              &inside_mesh, &hit_mesh, &distance );

            // If already inside the mesh
            if ( inside_mesh )
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_X2 );
                *geometry_distance = 0.0;
                return;
            }
            else if ( hit_mesh ) // Outside and hit the mesh
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_X2 );
                *geometry_distance = distance;
                return;
            }
            else // Not inside not hitting (then get the AABB distance)
            {
                *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_NOTHING );
                *geometry_distance = hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 1 ] );
                return;
            }
        }

        if ( in_obj == IN_JAW_Y1 )
        {
            m_transport_mesh( pos, dir, linac.Y_jaw_v1, linac.Y_jaw_v2, linac.Y_jaw_v3,
                              linac.Y_jaw_index[ 0 ], linac.Y_jaw_nb_triangles[ 0 ], geom_tol,
                              &inside_mesh, &hit_mesh, &distance );

            // If already inside the mesh
            if ( inside_mesh )
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_Y1 );
                *geometry_distance = 0.0;
                return;
            }
            else if ( hit_mesh ) // Outside and hit the mesh
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_Y1 );
                *geometry_distance = distance;
                return;
            }
            else // Not inside not hitting (then get the AABB distance)
            {
                *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_NOTHING );
                *geometry_distance = hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 0 ] );
                return;
            }
        }

        if ( in_obj == IN_JAW_Y2 )
        {
            m_transport_mesh( pos, dir, linac.Y_jaw_v1, linac.Y_jaw_v2, linac.Y_jaw_v3,
                              linac.Y_jaw_index[ 1 ], linac.Y_jaw_nb_triangles[ 1 ], geom_tol,
                              &inside_mesh, &hit_mesh, &distance );

            // If already inside the mesh
            if ( inside_mesh )
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_Y2 );
                *geometry_distance = 0.0;
                return;
            }
            else if ( hit_mesh ) // Outside and hit the mesh
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_JAW_Y2 );
                *geometry_distance = distance;
                return;
            }
            else // Not inside not hitting (then get the AABB distance)
            {
                *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_NOTHING );
                *geometry_distance = hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 1 ] );
                return;
            }
        }

        if ( in_obj == IN_BANK_A )
        {
            // Loop over leaves
            ileaf = 0; while( ileaf < linac.A_nb_leaves )
            {
                // If hit a leaf bounding box
                if ( test_ray_AABB( pos, dir, linac.A_leaf_aabb[ ileaf ] ) )
                {

                    m_transport_mesh( pos, dir, linac.A_leaf_v1, linac.A_leaf_v2, linac.A_leaf_v3,
                                      linac.A_leaf_index[ ileaf ], linac.A_leaf_nb_triangles[ ileaf ], geom_tol,
                                      &inside_mesh, &hit_mesh, &distance );

                    // If already inside of one of them
                    if ( inside_mesh )
                    {
                        *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                        *geometry_id = m_write_geom_type( *geometry_id, IN_BANK_A );
                        *geometry_id = m_write_geom_index( *geometry_id, ileaf );
                        *geometry_distance = 0.0;
                        return;
                    }
                    else if ( hit_mesh )
                    {
                        // Select the closest
                        if ( distance < *geometry_distance )
                        {
                            *geometry_distance = distance;
                            geom_index = ileaf;
                        }
                    }

                } // in a leaf bounding box

                ++ileaf;

            } // each leaf

            // No leaves were hit
            if ( geom_index < 0 )
            {
                *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_NOTHING );
                *geometry_id = m_write_geom_index( *geometry_id, 0 );
                *geometry_distance = hit_ray_AABB( pos, dir, linac.A_bank_aabb ); // Bounding box
            }
            else
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_BANK_A );
                *geometry_id = m_write_geom_index( *geometry_id, ui16( geom_index ) );
            }

            return;
        }

        if ( in_obj == IN_BANK_B )
        {
            // Loop over leaves
            ileaf = 0; while( ileaf < linac.B_nb_leaves )
            {
                // If hit a leaf bounding box
                if ( test_ray_AABB( pos, dir, linac.B_leaf_aabb[ ileaf ] ) )
                {

                    m_transport_mesh( pos, dir, linac.B_leaf_v1, linac.B_leaf_v2, linac.B_leaf_v3,
                                      linac.B_leaf_index[ ileaf ], linac.B_leaf_nb_triangles[ ileaf ], geom_tol,
                                      &inside_mesh, &hit_mesh, &distance );

                    // If already inside of one of them
                    if ( inside_mesh )
                    {
                        *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                        *geometry_id = m_write_geom_type( *geometry_id, IN_BANK_B );
                        *geometry_id = m_write_geom_index( *geometry_id, ileaf );
                        *geometry_distance = 0.0;
                        return;
                    }
                    else if ( hit_mesh )
                    {
                        // Select the closest
                        if ( distance < *geometry_distance )
                        {
                            *geometry_distance = distance;
                            geom_index = ileaf;
                        }
                    }

                } // in a leaf bounding box

                ++ileaf;

            } // each leaf

            // No leaves were hit
            if ( geom_index < 0 )
            {
                *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_NOTHING );
                *geometry_id = m_write_geom_index( *geometry_id, 0 );
                *geometry_distance = hit_ray_AABB( pos, dir, linac.B_bank_aabb ); // Bounding box
            }
            else
            {
                *geometry_id = m_write_geom_nav( *geometry_id, INSIDE_MESH );
                *geometry_id = m_write_geom_type( *geometry_id, IN_BANK_B );
                *geometry_id = m_write_geom_index( *geometry_id, ui16( geom_index ) );
            }

            return;
        }

    }


    // Should never reach here
#ifdef DEBUG
    printf("MLC navigation error: out of geometry\n");
#endif

}

__host__ __device__ void m_mlc_nav_in_mesh( f32xyz pos, f32xyz dir, LinacData linac,
                                            f32 geom_tol,
                                            ui32 *geometry_id, f32 *geometry_distance )
{

    //*geometry_distance = FLT_MAX;
    //*geometry_id = 0;
    //i8 navigation = OUTSIDE_MESH;



    // Read the geometry
    ui16 in_obj = m_read_geom_type( *geometry_id );

    bool inside_mesh = false;
    bool hit_mesh = false;
    f32 distance;

//    printf(" ::: Nav Inside in obj %i\n", in_obj);

    if ( in_obj == IN_JAW_X1 )
    {
        m_transport_mesh( pos, dir, linac.X_jaw_v1, linac.X_jaw_v2, linac.X_jaw_v3,
                          linac.X_jaw_index[ 0 ], linac.X_jaw_nb_triangles[ 0 ], geom_tol,
                          &inside_mesh, &hit_mesh, &distance );

        // If not inside (in case of crossing a tiny piece of matter get the AABB distance)
        *geometry_distance = ( inside_mesh ) ? distance : hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 0 ] );
    }

    else if ( in_obj == IN_JAW_X2 )
    {
        m_transport_mesh( pos, dir, linac.X_jaw_v1, linac.X_jaw_v2, linac.X_jaw_v3,
                          linac.X_jaw_index[ 1 ], linac.X_jaw_nb_triangles[ 1 ], geom_tol,
                          &inside_mesh, &hit_mesh, &distance );

        // If not inside (in case of crossing a tiny piece of matter get the AABB distance)
        *geometry_distance = ( inside_mesh ) ? distance : hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 1 ] );
    }

    else if ( in_obj == IN_JAW_Y1 )
    {
        m_transport_mesh( pos, dir, linac.Y_jaw_v1, linac.Y_jaw_v2, linac.Y_jaw_v3,
                          linac.Y_jaw_index[ 0 ], linac.Y_jaw_nb_triangles[ 0 ], geom_tol,
                          &inside_mesh, &hit_mesh, &distance );

        // If not inside (in case of crossing a tiny piece of matter get the AABB distance)
        *geometry_distance = ( inside_mesh ) ? distance : hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 0 ] );
    }

    else if ( in_obj == IN_JAW_Y2 )
    {
        m_transport_mesh( pos, dir, linac.Y_jaw_v1, linac.Y_jaw_v2, linac.Y_jaw_v3,
                          linac.Y_jaw_index[ 1 ], linac.Y_jaw_nb_triangles[ 1 ], geom_tol,
                          &inside_mesh, &hit_mesh, &distance );

        // If not inside (in case of crossing a tiny piece of matter get the AABB distance)
        *geometry_distance = ( inside_mesh ) ? distance : hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 1 ] );
    }

    else if ( in_obj == IN_BANK_A )
    {
        ui16 ileaf = m_read_geom_index( *geometry_id );

        m_transport_mesh( pos, dir, linac.A_leaf_v1, linac.A_leaf_v2, linac.A_leaf_v3,
                          linac.A_leaf_index[ ileaf ], linac.A_leaf_nb_triangles[ ileaf ], geom_tol,
                          &inside_mesh, &hit_mesh, &distance );

        // If not inside (in case of crossing a tiny piece of matter get the AABB distance)
        *geometry_distance = ( inside_mesh ) ? distance : hit_ray_AABB( pos, dir, linac.A_leaf_aabb[ ileaf ] );
    }

    else if ( in_obj == IN_BANK_B )
    {
        ui16 ileaf = m_read_geom_index( *geometry_id );

        m_transport_mesh( pos, dir, linac.B_leaf_v1, linac.B_leaf_v2, linac.B_leaf_v3,
                          linac.B_leaf_index[ ileaf ], linac.B_leaf_nb_triangles[ ileaf ], geom_tol,
                          &inside_mesh, &hit_mesh, &distance );

        // If not inside (in case of crossing a tiny piece of matter get the AABB distance)
        *geometry_distance = ( inside_mesh ) ? distance : hit_ray_AABB( pos, dir, linac.B_leaf_aabb[ ileaf ] );
    }

    else
    {
        // Should never reach here
        #ifdef DEBUG
            printf("MLC navigation error: out of geometry\n");
        #endif
    }

    *geometry_id = m_write_geom_nav( *geometry_id, OUTSIDE_MESH );
    *geometry_id = m_write_geom_type( *geometry_id, IN_NOTHING );
    return;

}


__host__ __device__ void MPLINACN::track_to_out( ParticlesData &particles, LinacData linac,
                                                 MaterialsTable materials,
                                                 PhotonCrossSectionTable photon_CS_table,
                                                 GlobalSimulationParametersData parameters,
                                                 ui32 id )
{
    // Read position
    f32xyz pos;
    pos.x = particles.px[ id ];
    pos.y = particles.py[ id ];
    pos.z = particles.pz[ id ];

    // Read direction
    f32xyz dir;
    dir.x = particles.dx[ id ];
    dir.y = particles.dy[ id ];
    dir.z = particles.dz[ id ];

    // In a mesh?
    ui8 navigation = m_read_geom_nav( particles.geometry_id[ id ] );

    //// Get material //////////////////////////////////////////////////////////////////

    i16 mat_id = ( navigation == INSIDE_MESH ) ? 0 : -1;   // 0 MLC mat, -1 not mat around the LINAC (vacuum)

//    if (id==92) printf("id %i  - mat id %i - navigation %i\n", id, mat_id, navigation);

    //// Find next discrete interaction ///////////////////////////////////////

    f32 next_interaction_distance = F32_MAX;
    ui8 next_discrete_process = 0;

    // If inside a mesh do physics else only tranportation (vacuum around the LINAC)
    if ( mat_id != - 1 )
    {
        photon_get_next_interaction ( particles, parameters, photon_CS_table, mat_id, id );
        next_interaction_distance = particles.next_interaction_distance[ id ];
        next_discrete_process = particles.next_discrete_process[ id ];
    }

    /// Get the hit distance of the closest geometry //////////////////////////////////

    f32 boundary_distance;
    ui32 next_geometry_id = particles.geometry_id[ id ];

    if ( navigation == INSIDE_MESH )
    {
//        if (id==92) printf("id %i - inside mesh - in obj %i\n", id, m_read_geom_type( next_geometry_id ));
        m_mlc_nav_in_mesh( pos, dir, linac, parameters.geom_tolerance, &next_geometry_id, &boundary_distance );
//        if (id==92) printf("id %i - inside mesh - dist %f\n", id, boundary_distance );

    }
    else
    {
        m_mlc_nav_out_mesh( pos, dir, linac, parameters.geom_tolerance, &next_geometry_id, &boundary_distance );
//        if (id==92) printf("id %i - outside mesh dist %f - hit obj %i\n", id, boundary_distance, m_read_geom_type( next_geometry_id ) );
    }

    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters.geom_tolerance; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

//    if (id==92) printf( "id %i cur pos %f %f %f next dist %f\n", id, pos.x, pos.y, pos.z, next_interaction_distance );

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // update tof
    //particles.tof[part_id] += c_light * next_interaction_distance;

    // store new position
    particles.px[ id ] = pos.x;
    particles.py[ id ] = pos.y;
    particles.pz[ id ] = pos.z;

//    printf( "id %i pos %f %f %f - aabb %f %f %f %f %f %f\n", id, pos.x, pos.y, pos.z,
//            linac.aabb.xmin, linac.aabb.xmax, linac.aabb.ymin, linac.aabb.ymax,
//            linac.aabb.zmin, linac.aabb.zmax );

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance ( pos, linac.aabb, parameters.geom_tolerance ) )
    {
        particles.endsimu[ id ] = PARTICLE_FREEZE;
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    if ( next_discrete_process != GEOMETRY_BOUNDARY )
    {
//        printf(" ---# phys effect\n");

//        if (id==92) printf("id %i phys effect\n", id);

        // Resolve discrete process
        SecParticle electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                                 materials, mat_id, id );

        //// Here e- are not tracked, and lost energy not drop
        //// Energy cut
        if ( particles.E[ id ] <= materials.photon_energy_cut[ mat_id ])
        {
            // kill without mercy (energy not drop)
            particles.endsimu[ id ] = PARTICLE_DEAD;
            return;
        }
    }
    else
    {
//        printf(" ---# geom effect\n");

//        if (id==92) printf("id %i geom effect\n", id);

        // Update geometry id
        particles.geometry_id[ id ] = next_geometry_id;
    }

    // DEBUG
    //particles.endsimu[ id ] = PARTICLE_DEAD;

}

__host__ __device__ void MPLINACN::track_to_out_nonav( ParticlesData &particles, LinacData linac,
                                                       ui32 id )
{
    // Read position
    f32xyz pos;
    pos.x = particles.px[ id ];
    pos.y = particles.py[ id ];
    pos.z = particles.pz[ id ];

    // Read direction
    f32xyz dir;
    dir.x = particles.dx[ id ];
    dir.y = particles.dy[ id ];
    dir.z = particles.dz[ id ];

//    printf("%i - pos %f %f %f  status %i\n", id, pos.x, pos.y, pos.z, particles.endsimu[ id ]);

    /// Get the hit distance of the closest geometry //////////////////////////////////

    ui16 in_obj = HIT_NOTHING;
    ui32 itri, offset, ileaf;
    f32 geom_distance;
    f32 min_distance = FLT_MAX;

    // First get the distance to the bounding box

    if ( linac.X_nb_jaw != 0 )
    {
        geom_distance = hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 0 ] );
        if ( geom_distance < min_distance )
        {
            min_distance = geom_distance;
            in_obj = HIT_JAW_X1;
        }

        geom_distance = hit_ray_AABB( pos, dir, linac.X_jaw_aabb[ 1 ] );
        if ( geom_distance < min_distance )
        {
            min_distance = geom_distance;
            in_obj = HIT_JAW_X2;
        }
    }

    if ( linac.Y_nb_jaw != 0 )
    {
        geom_distance = hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 0 ] );
        if ( geom_distance < min_distance )
        {
            min_distance = geom_distance;
            in_obj = HIT_JAW_Y1;
        }

        geom_distance = hit_ray_AABB( pos, dir, linac.Y_jaw_aabb[ 1 ] );
        if ( geom_distance < min_distance )
        {
            min_distance = geom_distance;
            in_obj = HIT_JAW_Y2;
        }
    }

    geom_distance = hit_ray_AABB( pos, dir, linac.A_bank_aabb );
    if ( geom_distance < min_distance )
    {
        min_distance = geom_distance;
        in_obj = HIT_BANK_A;
    }

    geom_distance = hit_ray_AABB( pos, dir, linac.B_bank_aabb );
    if ( geom_distance < min_distance )
    {
        min_distance = geom_distance;
        in_obj = HIT_BANK_B;
    }

    // Then check the distance by looking the complete mesh

    if ( in_obj == HIT_JAW_X1 )
    {
        in_obj = HIT_NOTHING;
        min_distance = FLT_MAX;

        itri = 0; while ( itri < linac.X_jaw_nb_triangles[ 0 ] )
        {
            offset = linac.X_jaw_index[ 0 ];
            geom_distance = hit_ray_triangle( pos, dir,
                                              linac.X_jaw_v1[ offset+itri ],
                    linac.X_jaw_v2[ offset+itri ],
                    linac.X_jaw_v3[ offset+itri ] );
            if ( geom_distance < min_distance )
            {
                geom_distance = min_distance;
                in_obj = HIT_JAW_X1;
            }
            ++itri;
        }
    }
    else if ( in_obj == HIT_JAW_X2 )
    {
        in_obj = HIT_NOTHING;
        min_distance = FLT_MAX;

        itri = 0; while ( itri < linac.X_jaw_nb_triangles[ 1 ] )
        {
            offset = linac.X_jaw_index[ 1 ];
            geom_distance = hit_ray_triangle( pos, dir,
                                              linac.X_jaw_v1[ offset+itri ],
                                              linac.X_jaw_v2[ offset+itri ],
                                              linac.X_jaw_v3[ offset+itri ] );
            if ( geom_distance < min_distance )
            {
                geom_distance = min_distance;
                in_obj = HIT_JAW_X2;
            }
            ++itri;
        }
    }
    else if ( in_obj == HIT_JAW_Y1 )
    {
        in_obj = HIT_NOTHING;
        min_distance = FLT_MAX;

        // Loop over triangles
        itri = 0; while ( itri < linac.Y_jaw_nb_triangles[ 0 ] )
        {
            offset = linac.Y_jaw_index[ 0 ];
            geom_distance = hit_ray_triangle( pos, dir,
                                              linac.Y_jaw_v1[ offset+itri ],
                                              linac.Y_jaw_v2[ offset+itri ],
                                              linac.Y_jaw_v3[ offset+itri ] );
            if ( geom_distance < min_distance )
            {
                geom_distance = min_distance;
                in_obj = HIT_JAW_Y1;
            }
            ++itri;
        }
    }
    else if ( in_obj == HIT_JAW_Y2 )
    {
        in_obj = HIT_NOTHING;
        min_distance = FLT_MAX;

        itri = 0; while ( itri < linac.Y_jaw_nb_triangles[ 1 ] )
        {
            offset = linac.Y_jaw_index[ 1 ];
            geom_distance = hit_ray_triangle( pos, dir,
                                              linac.Y_jaw_v1[ offset+itri ],
                                              linac.Y_jaw_v2[ offset+itri ],
                                              linac.Y_jaw_v3[ offset+itri ] );
            if ( geom_distance < min_distance )
            {
                geom_distance = min_distance;
                in_obj = HIT_JAW_Y2;
            }
            ++itri;
        }
    }
    else if ( in_obj == HIT_BANK_A )
    {
        in_obj = HIT_NOTHING;
        min_distance = FLT_MAX;

        ileaf = 0; while( ileaf < linac.A_nb_leaves )
        {
            // If hit a leaf
            if ( test_ray_AABB( pos, dir, linac.A_leaf_aabb[ ileaf ] ) )
            {
                // Loop over triangles
                itri = 0; while ( itri < linac.A_leaf_nb_triangles[ ileaf ] )
                {
                    offset = linac.A_leaf_index[ ileaf ];
                    geom_distance = hit_ray_triangle( pos, dir,
                                                      linac.A_leaf_v1[ offset+itri ],
                                                      linac.A_leaf_v2[ offset+itri ],
                                                      linac.A_leaf_v3[ offset+itri ] );
                    if ( geom_distance < min_distance )
                    {
                        geom_distance = min_distance;
                        in_obj = HIT_BANK_A;
                        //in_leaf = ileaf;
                    }
                    ++itri;
                }
            } // in a leaf bounding box

            ++ileaf;

        } // each leaf
    }
    else if ( in_obj == HIT_BANK_B )
    {
        in_obj = HIT_NOTHING;
        min_distance = FLT_MAX;

        ileaf = 0; while( ileaf < linac.B_nb_leaves )
        {
            // If hit a leaf
            if ( test_ray_AABB( pos, dir, linac.B_leaf_aabb[ ileaf ] ) )
            {
                // Loop over triangles
                itri = 0; while ( itri < linac.B_leaf_nb_triangles[ ileaf ] )
                {
                    offset = linac.B_leaf_index[ ileaf ];
                    geom_distance = hit_ray_triangle( pos, dir,
                                                      linac.B_leaf_v1[ offset+itri ],
                                                      linac.B_leaf_v2[ offset+itri ],
                                                      linac.B_leaf_v3[ offset+itri ] );
                    if ( geom_distance < min_distance )
                    {
                        geom_distance = min_distance;
                        in_obj = HIT_BANK_B;
                        //in_leaf = ileaf;
                    }
                    ++itri;
                }
            } // in a leaf bounding box

            ++ileaf;
        }
    }

    /// TODO - Navigation within element - Kill the particle without mercy

    if ( in_obj != HIT_NOTHING )
    {
        particles.endsimu[ id ] = PARTICLE_DEAD;
//        printf("%i kill touch %i\n", id, in_obj);
    }
    else
    {
        particles.endsimu[ id ] = PARTICLE_FREEZE;
//        printf("%i freeze touch %i\n", id, in_obj);
    }

}


// Device kernel that track particles within the voxelized volume until boundary
__global__ void MPLINACN::kernel_device_track_to_out( ParticlesData particles, LinacData linac,
                                                      MaterialsTable materials, PhotonCrossSectionTable photon_CS,
                                                      GlobalSimulationParametersData parameters,
                                                      bool nav_within_mlc )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    // Init geometry ID for navigation
    particles.geometry_id[ id ] = 0;

    // Stepping loop
    if ( nav_within_mlc )
    {
        // DEBUG
        ui32 i = 0;

        while ( particles.endsimu[ id ] != PARTICLE_DEAD && particles.endsimu[ id ] != PARTICLE_FREEZE )
        {
            //printf("Step\n");
            MPLINACN::track_to_out( particles, linac, materials, photon_CS, parameters, id );

            if ( i > 100 )
            {
                printf(" ID %i break loop\n", id );
                break;
            }

            ++i;
        }
    }
    else
    {
        while ( particles.endsimu[ id ] != PARTICLE_DEAD && particles.endsimu[ id ] != PARTICLE_FREEZE )
        {
            MPLINACN::track_to_out_nonav( particles, linac, id );
        }
    }

    /// Move the particle back to the global frame ///

    // read position and direction
    f32xyz pos = make_f32xyz( particles.px[ id ], particles.py[ id ], particles.pz[ id ] );
    f32xyz dir = make_f32xyz( particles.dx[ id ], particles.dy[ id ], particles.dz[ id ] );

    // Change the frame to the particle (global to linac)
    pos = fxyz_local_to_global_position( linac.transform, pos );
    dir = fxyz_local_to_global_direction( linac.transform, dir );

    // Store data
    particles.px[ id ] = pos.x;
    particles.py[ id ] = pos.y;
    particles.pz[ id ] = pos.z;
    particles.dx[ id ] = dir.x;
    particles.dy[ id ] = dir.y;
    particles.dz[ id ] = dir.z;

}

// Host kernel that track particles within the voxelized volume until boundary
void MPLINACN::kernel_host_track_to_out( ParticlesData particles, LinacData linac,
                                         MaterialsTable materials, PhotonCrossSectionTable photon_CS,
                                         GlobalSimulationParametersData parameters,
                                         bool nav_within_mlc, ui32 id )
{
    // Init geometry ID for navigation
    particles.geometry_id[ id ] = 0;

    // Stepping loop
    if ( nav_within_mlc )
    {
        while ( particles.endsimu[ id ] != PARTICLE_DEAD && particles.endsimu[ id ] != PARTICLE_FREEZE )
        {
            MPLINACN::track_to_out( particles, linac, materials, photon_CS, parameters, id );
        }
    }
    else
    {
        while ( particles.endsimu[ id ] != PARTICLE_DEAD && particles.endsimu[ id ] != PARTICLE_FREEZE )
        {
            MPLINACN::track_to_out_nonav( particles, linac, id );
        }
    }
    /// Move the particle back to the global frame ///

    // read position and direction
    f32xyz pos = make_f32xyz( particles.px[ id ], particles.py[ id ], particles.pz[ id ] );
    f32xyz dir = make_f32xyz( particles.dx[ id ], particles.dy[ id ], particles.dz[ id ] );

    // Change the frame to the particle (global to linac)
    pos = fxyz_local_to_global_position( linac.transform, pos );
    dir = fxyz_local_to_global_direction( linac.transform, dir );

    // Store data
    particles.px[ id ] = pos.x;
    particles.py[ id ] = pos.y;
    particles.pz[ id ] = pos.z;
    particles.dx[ id ] = dir.x;
    particles.dy[ id ] = dir.y;
    particles.dz[ id ] = dir.z;

}

////// Privates /////////////////////////////////////////////////////////////////////////////

// Read the list of tokens in a txt line
std::vector< std::string > MeshPhanLINACNav::m_split_txt( std::string line ) {

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));

    return tokens;

}

void MeshPhanLINACNav::m_init_mlc()
{
    // First check the file
    std::string ext = m_mlc_filename.substr( m_mlc_filename.find_last_of( "." ) + 1 );
    if ( ext != "obj" )
    {
        GGcerr << "MeshPhanLINACNav can only read mesh data in Wavefront format (.obj)!" << GGendl;
        exit_simulation();
    }

    // Then get data
    MeshIO *meshio = new MeshIO;
    MeshData mlc = meshio->read_mesh_file( m_mlc_filename );

//    ui32 i = 0; while ( i < mlc.mesh_names.size() )
//    {
//        GGcout << "Mesh " << i << GGendl;
//        ui32 offset = mlc.mesh_index[ i ];

//        ui32 j = 0; while ( j < mlc.nb_triangles[ i ] )
//        {
//            ui32 ii = offset+j;
//            printf("  %f %f %f  -  %f %f %f  -  %f %f %f\n", mlc.v1[ii].x, mlc.v1[ii].y, mlc.v1[ii].z,
//                   mlc.v2[ii].x, mlc.v2[ii].y, mlc.v2[ii].z,
//                   mlc.v3[ii].x, mlc.v3[ii].y, mlc.v3[ii].z );
//            ++j;
//        }

//        ++i;
//    }

//    GGcout << "Meshes read" << GGendl;

    // Check if there are at least one leaf
    if ( mlc.mesh_names.size() == 0 )
    {
        GGcerr << "MeshPhanLINACNav, no leaves in the mlc file were found!" << GGendl;
        exit_simulation();
    }

    // Check if the number of leaves match with the provided parameters
    if ( m_linac.A_nb_leaves + m_linac.B_nb_leaves !=  mlc.mesh_names.size() )
    {
        GGcerr << "MeshPhanLINACNav, number of leaves provided by the user is different to the number of meshes contained on the file!" << GGendl;
        exit_simulation();
    }

    // Some allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_index), m_linac.A_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_nb_triangles), m_linac.A_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_aabb), m_linac.A_nb_leaves * sizeof( AabbData ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_index), m_linac.B_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_nb_triangles), m_linac.B_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_aabb), m_linac.B_nb_leaves * sizeof( AabbData ) ) );

//    GGcout << "first allocation" << GGendl;

    // Pre-calculation and checking of the data
    ui32 i_leaf = 0;
    std::string leaf_name, bank_name;
    ui32 index_leaf_bank;
    ui32 tot_tri_bank_A = 0;
    ui32 tot_tri_bank_B = 0;

    while ( i_leaf < mlc.mesh_names.size() )
    {
        // Get name of the leaf
        leaf_name = mlc.mesh_names[ i_leaf ];

        // Bank A or B
        bank_name = leaf_name[ 0 ];

        // Check
        if ( bank_name != "A" && bank_name != "B" )
        {
            GGcerr << "MeshPhanLINACNav: name of each leaf must start by the bank 'A' or 'B', " << bank_name << " given!" << GGendl;
            exit_simulation();
        }

        // Get leaf index
        index_leaf_bank = std::stoi( leaf_name.substr( 1, leaf_name.size()-1 ) );

        // If bank A
        if ( bank_name == "A" )
        {
            // Check
            if ( index_leaf_bank == 0 || index_leaf_bank > m_linac.A_nb_leaves )
            {
                GGcerr << "MeshPhanLINACNav: name of leaves must have index starting from 1 to N leaves!" << GGendl;
                exit_simulation();
            }

            // Store in sort way te number of triangles for each leaf
            // index_leaf_bank-1 because leaf start from 1 to N
            m_linac.A_leaf_nb_triangles[ index_leaf_bank-1 ] = mlc.nb_triangles[ i_leaf ];
            tot_tri_bank_A += mlc.nb_triangles[ i_leaf ];

//            GGcout << " A nb tri " << m_linac.A_leaf_nb_triangles[ index_leaf_bank-1 ] << " ileaf " << i_leaf << GGendl;

        }

        // If bank B
        if ( bank_name == "B" )
        {
            // Check
            if ( index_leaf_bank == 0 || index_leaf_bank > m_linac.B_nb_leaves )
            {
                GGcerr << "MeshPhanLINACNav: name of leaves must have index starting from 1 to N leaves!" << GGendl;
                exit_simulation();
            }

            // Store in sort way te number of triangles for each leaf
            // index_leaf_bank-1 because leaf start from 1 to N
            m_linac.B_leaf_nb_triangles[ index_leaf_bank-1 ] = mlc.nb_triangles[ i_leaf ];
            tot_tri_bank_B += mlc.nb_triangles[ i_leaf ];

//            GGcout << " B nb tri " << m_linac.B_leaf_nb_triangles[ index_leaf_bank-1 ] << " ileaf " << i_leaf << GGendl;
        }

        ++i_leaf;
    } // i_leaf

//    GGcout << "Check ok" << GGendl;

    // Compute the offset for each leaf from bank A
    m_linac.A_leaf_index[ 0 ] = 0;
    i_leaf = 1; while ( i_leaf < m_linac.A_nb_leaves )
    {
        m_linac.A_leaf_index[ i_leaf ] = m_linac.A_leaf_index[ i_leaf-1 ] + m_linac.A_leaf_nb_triangles[ i_leaf-1 ];

//        GGcout << " A offset " << m_linac.A_leaf_index[ i_leaf ]
//                  << " ileaf " << i_leaf << " nb tri: " << m_linac.A_leaf_nb_triangles[ i_leaf ] << GGendl;

        ++i_leaf;

    }

    // Compute the offset for each leaf from bank B
    m_linac.B_leaf_index[ 0 ] = 0;
    i_leaf = 1; while ( i_leaf < m_linac.B_nb_leaves )
    {
        m_linac.B_leaf_index[ i_leaf ] = m_linac.B_leaf_index[ i_leaf-1 ] + m_linac.B_leaf_nb_triangles[ i_leaf-1 ];
//        GGcout << " B offset " << m_linac.B_leaf_index[ i_leaf ]
//                  << " ileaf " << i_leaf << " nb tri: " << m_linac.B_leaf_nb_triangles[ i_leaf ] << GGendl;

        ++i_leaf;
    }

//    GGcout << "Get offset" << GGendl;

    // Some others allocations
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_v1), tot_tri_bank_A * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_v2), tot_tri_bank_A * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_v3), tot_tri_bank_A * sizeof( f32xyz ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_v1), tot_tri_bank_B * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_v2), tot_tri_bank_B * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_v3), tot_tri_bank_B * sizeof( f32xyz ) ) );

//    GGcout << "Second allocation" << GGendl;

    // Loop over leaf. Organize mesh data into the linac data.
    ui32 i_tri, offset_bank, offset_mlc;
    f32xyz v1, v2, v3;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;
    i_leaf = 0; while ( i_leaf < mlc.mesh_names.size() )
    {
        // Get name of the leaf
        leaf_name = mlc.mesh_names[ i_leaf ];

        // Bank A or B
        bank_name = leaf_name[ 0 ];

        // Get leaf index within the bank
        index_leaf_bank = std::stoi( leaf_name.substr( 1, leaf_name.size()-1 ) ) - 1; // -1 because leaf start from 1 to N

        // index within the mlc (all meshes)
        offset_mlc = mlc.mesh_index[ i_leaf ];

//        GGcout << "leaf " << i_leaf << " name: " << leaf_name
//               << " bank: " << bank_name << " index: " << index_leaf_bank
//               << " offset: " << offset_mlc << GGendl;

        // Init AABB
        xmin = FLT_MAX; xmax = -FLT_MAX;
        ymin = FLT_MAX; ymax = -FLT_MAX;
        zmin = FLT_MAX; zmax = -FLT_MAX;

        // If bank A
        if ( bank_name == "A" )
        {
            // index within the bank
            offset_bank = m_linac.A_leaf_index[ index_leaf_bank ];

//            GGcout << "    A offset bank: " << offset_bank << GGendl;

//            GGcout << " Bank A leaft " << index_leaf_bank << GGendl;

            // loop over triangles
            i_tri = 0; while ( i_tri < m_linac.A_leaf_nb_triangles[ index_leaf_bank ] )
            {
                // Store on the right place
                v1 = mlc.v1[ offset_mlc + i_tri ];
                v2 = mlc.v2[ offset_mlc + i_tri ];
                v3 = mlc.v3[ offset_mlc + i_tri ];

//                if ( index_leaf_bank == 0 )
//                {
//                    printf("  v1 %f %f %f # v2 %f %f %f # v3 %f %f %f\n", v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
//                }

                m_linac.A_leaf_v1[ offset_bank + i_tri ] = v1;
                m_linac.A_leaf_v2[ offset_bank + i_tri ] = v2;
                m_linac.A_leaf_v3[ offset_bank + i_tri ] = v3;

                // Determine AABB
                if ( v1.x > xmax ) xmax = v1.x;
                if ( v2.x > xmax ) xmax = v2.x;
                if ( v3.x > xmax ) xmax = v3.x;

                if ( v1.y > ymax ) ymax = v1.y;
                if ( v2.y > ymax ) ymax = v2.y;
                if ( v3.y > ymax ) ymax = v3.y;

                if ( v1.z > zmax ) zmax = v1.z;
                if ( v2.z > zmax ) zmax = v2.z;
                if ( v3.z > zmax ) zmax = v3.z;

                if ( v1.x < xmin ) xmin = v1.x;
                if ( v2.x < xmin ) xmin = v2.x;
                if ( v3.x < xmin ) xmin = v3.x;

                if ( v1.y < ymin ) ymin = v1.y;
                if ( v2.y < ymin ) ymin = v2.y;
                if ( v3.y < ymin ) ymin = v3.y;

                if ( v1.z < zmin ) zmin = v1.z;
                if ( v2.z < zmin ) zmin = v2.z;
                if ( v3.z < zmin ) zmin = v3.z;

                ++i_tri;
            }

//            GGcout << "    A tri process" << GGendl;

            // Store the bounding box of the current leaf
            m_linac.A_leaf_aabb[ index_leaf_bank ].xmin = xmin;
            m_linac.A_leaf_aabb[ index_leaf_bank ].xmax = xmax;
            m_linac.A_leaf_aabb[ index_leaf_bank ].ymin = ymin;
            m_linac.A_leaf_aabb[ index_leaf_bank ].ymax = ymax;
            m_linac.A_leaf_aabb[ index_leaf_bank ].zmin = zmin;
            m_linac.A_leaf_aabb[ index_leaf_bank ].zmax = zmax;

//            GGcout << "    A aabb" << GGendl;
//            GGcout << " A" << index_leaf_bank << " aabb: " << xmin << " " << xmax << GGendl;

        }
        else // Bank B
        {
            // index within the bank
            offset_bank = m_linac.B_leaf_index[ index_leaf_bank ];

//            GGcout << "    B offset bank: " << offset_bank << GGendl;

            // loop over triangles
            i_tri = 0; while ( i_tri < m_linac.B_leaf_nb_triangles[ index_leaf_bank ] )
            {
                // Store on the right place
                v1 = mlc.v1[ offset_mlc + i_tri ];
                v2 = mlc.v2[ offset_mlc + i_tri ];
                v3 = mlc.v3[ offset_mlc + i_tri ];

                m_linac.B_leaf_v1[ offset_bank + i_tri ] = v1;
                m_linac.B_leaf_v2[ offset_bank + i_tri ] = v2;
                m_linac.B_leaf_v3[ offset_bank + i_tri ] = v3;

                // Determine AABB
                if ( v1.x > xmax ) xmax = v1.x;
                if ( v2.x > xmax ) xmax = v2.x;
                if ( v3.x > xmax ) xmax = v3.x;

                if ( v1.y > ymax ) ymax = v1.y;
                if ( v2.y > ymax ) ymax = v2.y;
                if ( v3.y > ymax ) ymax = v3.y;

                if ( v1.z > zmax ) zmax = v1.z;
                if ( v2.z > zmax ) zmax = v2.z;
                if ( v3.z > zmax ) zmax = v3.z;

                if ( v1.x < xmin ) xmin = v1.x;
                if ( v2.x < xmin ) xmin = v2.x;
                if ( v3.x < xmin ) xmin = v3.x;

                if ( v1.y < ymin ) ymin = v1.y;
                if ( v2.y < ymin ) ymin = v2.y;
                if ( v3.y < ymin ) ymin = v3.y;

                if ( v1.z < zmin ) zmin = v1.z;
                if ( v2.z < zmin ) zmin = v2.z;
                if ( v3.z < zmin ) zmin = v3.z;

                ++i_tri;
            }

//            GGcout << "    B tri process" << GGendl;

            // Store the bounding box of the current leaf
            m_linac.B_leaf_aabb[ index_leaf_bank ].xmin = xmin;
            m_linac.B_leaf_aabb[ index_leaf_bank ].xmax = xmax;
            m_linac.B_leaf_aabb[ index_leaf_bank ].ymin = ymin;
            m_linac.B_leaf_aabb[ index_leaf_bank ].ymax = ymax;
            m_linac.B_leaf_aabb[ index_leaf_bank ].zmin = zmin;
            m_linac.B_leaf_aabb[ index_leaf_bank ].zmax = zmax;

//            GGcout << "    B aabb " << GGendl;
//            GGcout << " B" << index_leaf_bank << " aabb: " << xmin << " " << xmax << GGendl;
        }

        ++i_leaf;
    } // i_leaf

//    GGcout << "Organize data" << GGendl;

    // Finally, compute the AABB of the bank A
    xmin = FLT_MAX; xmax = -FLT_MAX;
    ymin = FLT_MAX; ymax = -FLT_MAX;
    zmin = FLT_MAX; zmax = -FLT_MAX;
    i_leaf = 0; while ( i_leaf < m_linac.A_nb_leaves )
    {
        if ( m_linac.A_leaf_aabb[ i_leaf ].xmin < xmin ) xmin = m_linac.A_leaf_aabb[ i_leaf ].xmin;
        if ( m_linac.A_leaf_aabb[ i_leaf ].ymin < ymin ) ymin = m_linac.A_leaf_aabb[ i_leaf ].ymin;
        if ( m_linac.A_leaf_aabb[ i_leaf ].zmin < zmin ) zmin = m_linac.A_leaf_aabb[ i_leaf ].zmin;

        if ( m_linac.A_leaf_aabb[ i_leaf ].xmax > xmax ) xmax = m_linac.A_leaf_aabb[ i_leaf ].xmax;
        if ( m_linac.A_leaf_aabb[ i_leaf ].ymax > ymax ) ymax = m_linac.A_leaf_aabb[ i_leaf ].ymax;
        if ( m_linac.A_leaf_aabb[ i_leaf ].zmax > zmax ) zmax = m_linac.A_leaf_aabb[ i_leaf ].zmax;

        ++i_leaf;
    }

    m_linac.A_bank_aabb.xmin = xmin;
    m_linac.A_bank_aabb.xmax = xmax;
    m_linac.A_bank_aabb.ymin = ymin;
    m_linac.A_bank_aabb.ymax = ymax;
    m_linac.A_bank_aabb.zmin = zmin;
    m_linac.A_bank_aabb.zmax = zmax;

    // And for the bank B
    xmin = FLT_MAX; xmax = -FLT_MAX;
    ymin = FLT_MAX; ymax = -FLT_MAX;
    zmin = FLT_MAX; zmax = -FLT_MAX;
    i_leaf = 0; while ( i_leaf < m_linac.B_nb_leaves )
    {
        if ( m_linac.B_leaf_aabb[ i_leaf ].xmin < xmin ) xmin = m_linac.B_leaf_aabb[ i_leaf ].xmin;
        if ( m_linac.B_leaf_aabb[ i_leaf ].ymin < ymin ) ymin = m_linac.B_leaf_aabb[ i_leaf ].ymin;
        if ( m_linac.B_leaf_aabb[ i_leaf ].zmin < zmin ) zmin = m_linac.B_leaf_aabb[ i_leaf ].zmin;

        if ( m_linac.B_leaf_aabb[ i_leaf ].xmax > xmax ) xmax = m_linac.B_leaf_aabb[ i_leaf ].xmax;
        if ( m_linac.B_leaf_aabb[ i_leaf ].ymax > ymax ) ymax = m_linac.B_leaf_aabb[ i_leaf ].ymax;
        if ( m_linac.B_leaf_aabb[ i_leaf ].zmax > zmax ) zmax = m_linac.B_leaf_aabb[ i_leaf ].zmax;

        ++i_leaf;
    }

    m_linac.B_bank_aabb.xmin = xmin;
    m_linac.B_bank_aabb.xmax = xmax;
    m_linac.B_bank_aabb.ymin = ymin;
    m_linac.B_bank_aabb.ymax = ymax;
    m_linac.B_bank_aabb.zmin = zmin;
    m_linac.B_bank_aabb.zmax = zmax;

//    GGcout << "Get AABB" << GGendl;

}


void MeshPhanLINACNav::m_init_jaw_x()
{
    // First check the file
    std::string ext = m_jaw_x_filename.substr( m_jaw_x_filename.find_last_of( "." ) + 1 );
    if ( ext != "obj" )
    {
        GGcerr << "MeshPhanLINACNav can only read mesh data in Wavefront format (.obj)!" << GGendl;
        exit_simulation();
    }

    // Then get data
    MeshIO *meshio = new MeshIO;
    MeshData jaw = meshio->read_mesh_file( m_jaw_x_filename );

    // Check if there are at least one jaw
    if ( jaw.mesh_names.size() == 0 )
    {
        GGcerr << "MeshPhanLINACNav, no jaw in the x-jaw file were found!" << GGendl;
        exit_simulation();
    }

    m_linac.X_nb_jaw = jaw.mesh_names.size();

    // Some allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_index), m_linac.X_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_nb_triangles), m_linac.X_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_aabb), m_linac.X_nb_jaw * sizeof( AabbData ) ) );

    // Pre-calculation and checking of the data
    ui32 i_jaw = 0;
    std::string jaw_name, axis_name;
    ui32 index_jaw;
    ui32 tot_tri_jaw = 0;

    while ( i_jaw < m_linac.X_nb_jaw )
    {
        // Get name of the jaw
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Name axis
        axis_name = jaw_name[ 0 ];

        // Check
        if ( axis_name != "X" )
        {
            GGcerr << "MeshPhanLINACNav: name of each jaw (in X) must start by 'X', " << axis_name << " given!" << GGendl;
            exit_simulation();
        }

        // Get leaf index
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) );

        // Check
        if ( index_jaw == 0 || index_jaw > 2 )
        {
            GGcerr << "MeshPhanLINACNav: name of jaws must have index starting from 1 to 2!" << GGendl;
            exit_simulation();
        }

        // Store the number of triangles for each jaw
        // index-1 because jaw start from 1 to 2
        m_linac.X_jaw_nb_triangles[ index_jaw-1 ] = jaw.nb_triangles[ i_jaw ];
        tot_tri_jaw += jaw.nb_triangles[ i_jaw ];

        ++i_jaw;
    } // i_leaf

    // Compute the offset for each jaw
    m_linac.X_jaw_index[ 0 ] = 0;
    m_linac.X_jaw_index[ 1 ] = m_linac.X_jaw_nb_triangles[ 0 ];

    // Some others allocations
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_v1), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_v2), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_v3), tot_tri_jaw * sizeof( f32xyz ) ) );

    // Loop over leaf. Organize mesh data into the linac data.
    ui32 i_tri, offset_mesh, offset_linac;
    f32xyz v1, v2, v3;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;
    i_jaw = 0; while ( i_jaw < m_linac.X_nb_jaw )
    {
        // Get name of the leaf
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Get leaf index within the bank
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) ) - 1; // -1 because jaw start from 1 to 2

        // index within the mlc (all meshes)
        offset_mesh = jaw.mesh_index[ i_jaw ];

        // Init AABB
        xmin = FLT_MAX; xmax = -FLT_MAX;
        ymin = FLT_MAX; ymax = -FLT_MAX;
        zmin = FLT_MAX; zmax = -FLT_MAX;

        // index within the bank
        offset_linac = m_linac.X_jaw_index[ index_jaw ];

        // loop over triangles
        i_tri = 0; while ( i_tri < m_linac.X_jaw_nb_triangles[ index_jaw ] )
        {
            // Store on the right place
            v1 = jaw.v1[ offset_mesh + i_tri ];
            v2 = jaw.v2[ offset_mesh + i_tri ];
            v3 = jaw.v3[ offset_mesh + i_tri ];

            m_linac.X_jaw_v1[ offset_linac + i_tri ] = v1;
            m_linac.X_jaw_v2[ offset_linac + i_tri ] = v2;
            m_linac.X_jaw_v3[ offset_linac + i_tri ] = v3;

            // Determine AABB
            if ( v1.x > xmax ) xmax = v1.x;
            if ( v2.x > xmax ) xmax = v2.x;
            if ( v3.x > xmax ) xmax = v3.x;

            if ( v1.y > ymax ) ymax = v1.y;
            if ( v2.y > ymax ) ymax = v2.y;
            if ( v3.y > ymax ) ymax = v3.y;

            if ( v1.z > zmax ) zmax = v1.z;
            if ( v2.z > zmax ) zmax = v2.z;
            if ( v3.z > zmax ) zmax = v3.z;

            if ( v1.x < xmin ) xmin = v1.x;
            if ( v2.x < xmin ) xmin = v2.x;
            if ( v3.x < xmin ) xmin = v3.x;

            if ( v1.y < ymin ) ymin = v1.y;
            if ( v2.y < ymin ) ymin = v2.y;
            if ( v3.y < ymin ) ymin = v3.y;

            if ( v1.z < zmin ) zmin = v1.z;
            if ( v2.z < zmin ) zmin = v2.z;
            if ( v3.z < zmin ) zmin = v3.z;

            ++i_tri;
        }

        // Store the bounding box of the current jaw
        m_linac.X_jaw_aabb[ index_jaw ].xmin = xmin;
        m_linac.X_jaw_aabb[ index_jaw ].xmax = xmax;
        m_linac.X_jaw_aabb[ index_jaw ].ymin = ymin;
        m_linac.X_jaw_aabb[ index_jaw ].ymax = ymax;
        m_linac.X_jaw_aabb[ index_jaw ].zmin = zmin;
        m_linac.X_jaw_aabb[ index_jaw ].zmax = zmax;

        ++i_jaw;
    } // i_jaw

}

void MeshPhanLINACNav::m_init_jaw_y()
{
    // First check the file
    std::string ext = m_jaw_y_filename.substr( m_jaw_y_filename.find_last_of( "." ) + 1 );
    if ( ext != "obj" )
    {
        GGcerr << "MeshPhanLINACNav can only read mesh data in Wavefront format (.obj)!" << GGendl;
        exit_simulation();
    }

    // Then get data
    MeshIO *meshio = new MeshIO;
    MeshData jaw = meshio->read_mesh_file( m_jaw_y_filename );

    // Check if there are at least one jaw
    if ( jaw.mesh_names.size() == 0 )
    {
        GGcerr << "MeshPhanLINACNav, no jaw in the y-jaw file were found!" << GGendl;
        exit_simulation();
    }

    m_linac.Y_nb_jaw = jaw.mesh_names.size();

    // Some allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_index), m_linac.Y_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_nb_triangles), m_linac.Y_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_aabb), m_linac.Y_nb_jaw * sizeof( AabbData ) ) );

    // Pre-calculation and checking of the data
    ui32 i_jaw = 0;
    std::string jaw_name, axis_name;
    ui32 index_jaw;
    ui32 tot_tri_jaw = 0;

    while ( i_jaw < m_linac.Y_nb_jaw )
    {
        // Get name of the jaw
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Name axis
        axis_name = jaw_name[ 0 ];

        // Check
        if ( axis_name != "Y" )
        {
            GGcerr << "MeshPhanLINACNav: name of each jaw (in Y) must start by 'Y', " << axis_name << " given!" << GGendl;
            exit_simulation();
        }

        // Get leaf index
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) );

        // Check
        if ( index_jaw == 0 || index_jaw > 2 )
        {
            GGcerr << "MeshPhanLINACNav: name of jaws must have index starting from 1 to 2!" << GGendl;
            exit_simulation();
        }

        // Store the number of triangles for each jaw
        // index-1 because jaw start from 1 to 2
        m_linac.Y_jaw_nb_triangles[ index_jaw-1 ] = jaw.nb_triangles[ i_jaw ];
        tot_tri_jaw += jaw.nb_triangles[ i_jaw ];

        ++i_jaw;
    } // i_leaf

    // Compute the offset for each jaw
    m_linac.Y_jaw_index[ 0 ] = 0;
    m_linac.Y_jaw_index[ 1 ] = m_linac.Y_jaw_nb_triangles[ 0 ];

    // Some others allocations
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_v1), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_v2), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_v3), tot_tri_jaw * sizeof( f32xyz ) ) );

    // Loop over leaf. Organize mesh data into the linac data.
    ui32 i_tri, offset_mesh, offset_linac;
    f32xyz v1, v2, v3;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;
    i_jaw = 0; while ( i_jaw < m_linac.Y_nb_jaw )
    {
        // Get name of the leaf
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Get leaf index within the bank
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) ) - 1; // -1 because jaw start from 1 to 2

        // index within the mlc (all meshes)
        offset_mesh = jaw.mesh_index[ i_jaw ];

        // Init AABB
        xmin = FLT_MAX; xmax = -FLT_MAX;
        ymin = FLT_MAX; ymax = -FLT_MAX;
        zmin = FLT_MAX; zmax = -FLT_MAX;

        // index within the bank
        offset_linac = m_linac.Y_jaw_index[ index_jaw ];

        // loop over triangles
        i_tri = 0; while ( i_tri < m_linac.Y_jaw_nb_triangles[ index_jaw ] )
        {
            // Store on the right place
            v1 = jaw.v1[ offset_mesh + i_tri ];
            v2 = jaw.v2[ offset_mesh + i_tri ];
            v3 = jaw.v3[ offset_mesh + i_tri ];

            m_linac.Y_jaw_v1[ offset_linac + i_tri ] = v1;
            m_linac.Y_jaw_v2[ offset_linac + i_tri ] = v2;
            m_linac.Y_jaw_v3[ offset_linac + i_tri ] = v3;

            // Determine AABB
            if ( v1.x > xmax ) xmax = v1.x;
            if ( v2.x > xmax ) xmax = v2.x;
            if ( v3.x > xmax ) xmax = v3.x;

            if ( v1.y > ymax ) ymax = v1.y;
            if ( v2.y > ymax ) ymax = v2.y;
            if ( v3.y > ymax ) ymax = v3.y;

            if ( v1.z > zmax ) zmax = v1.z;
            if ( v2.z > zmax ) zmax = v2.z;
            if ( v3.z > zmax ) zmax = v3.z;

            if ( v1.x < xmin ) xmin = v1.x;
            if ( v2.x < xmin ) xmin = v2.x;
            if ( v3.x < xmin ) xmin = v3.x;

            if ( v1.y < ymin ) ymin = v1.y;
            if ( v2.y < ymin ) ymin = v2.y;
            if ( v3.y < ymin ) ymin = v3.y;

            if ( v1.z < zmin ) zmin = v1.z;
            if ( v2.z < zmin ) zmin = v2.z;
            if ( v3.z < zmin ) zmin = v3.z;

            ++i_tri;
        }

        // Store the bounding box of the current jaw
        m_linac.Y_jaw_aabb[ index_jaw ].xmin = xmin;
        m_linac.Y_jaw_aabb[ index_jaw ].xmax = xmax;
        m_linac.Y_jaw_aabb[ index_jaw ].ymin = ymin;
        m_linac.Y_jaw_aabb[ index_jaw ].ymax = ymax;
        m_linac.Y_jaw_aabb[ index_jaw ].zmin = zmin;
        m_linac.Y_jaw_aabb[ index_jaw ].zmax = zmax;

        ++i_jaw;
    } // i_jaw

}

void MeshPhanLINACNav::m_translate_jaw_x( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.X_jaw_index[ index ];
    ui32 nb_tri = m_linac.X_jaw_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.X_jaw_v1[ offset + i_tri ] = fxyz_add( m_linac.X_jaw_v1[ offset + i_tri ], T );
        m_linac.X_jaw_v2[ offset + i_tri ] = fxyz_add( m_linac.X_jaw_v2[ offset + i_tri ], T );
        m_linac.X_jaw_v3[ offset + i_tri ] = fxyz_add( m_linac.X_jaw_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.X_jaw_aabb[ index ].xmin += T.x;
    m_linac.X_jaw_aabb[ index ].xmax += T.x;
    m_linac.X_jaw_aabb[ index ].ymin += T.y;
    m_linac.X_jaw_aabb[ index ].ymax += T.y;
    m_linac.X_jaw_aabb[ index ].zmin += T.z;
    m_linac.X_jaw_aabb[ index ].zmax += T.z;
}

void MeshPhanLINACNav::m_translate_jaw_y( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.Y_jaw_index[ index ];
    ui32 nb_tri = m_linac.Y_jaw_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.Y_jaw_v1[ offset + i_tri ] = fxyz_add( m_linac.Y_jaw_v1[ offset + i_tri ], T );
        m_linac.Y_jaw_v2[ offset + i_tri ] = fxyz_add( m_linac.Y_jaw_v2[ offset + i_tri ], T );
        m_linac.Y_jaw_v3[ offset + i_tri ] = fxyz_add( m_linac.Y_jaw_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.Y_jaw_aabb[ index ].xmin += T.x;
    m_linac.Y_jaw_aabb[ index ].xmax += T.x;
    m_linac.Y_jaw_aabb[ index ].ymin += T.y;
    m_linac.Y_jaw_aabb[ index ].ymax += T.y;
    m_linac.Y_jaw_aabb[ index ].zmin += T.z;
    m_linac.Y_jaw_aabb[ index ].zmax += T.z;
}

void MeshPhanLINACNav::m_translate_leaf_A( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.A_leaf_index[ index ];
    ui32 nb_tri = m_linac.A_leaf_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.A_leaf_v1[ offset + i_tri ] = fxyz_add( m_linac.A_leaf_v1[ offset + i_tri ], T );
        m_linac.A_leaf_v2[ offset + i_tri ] = fxyz_add( m_linac.A_leaf_v2[ offset + i_tri ], T );
        m_linac.A_leaf_v3[ offset + i_tri ] = fxyz_add( m_linac.A_leaf_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.A_leaf_aabb[ index ].xmin += T.x;
    m_linac.A_leaf_aabb[ index ].xmax += T.x;
    m_linac.A_leaf_aabb[ index ].ymin += T.y;
    m_linac.A_leaf_aabb[ index ].ymax += T.y;
    m_linac.A_leaf_aabb[ index ].zmin += T.z;
    m_linac.A_leaf_aabb[ index ].zmax += T.z;

    // Update the bank AABB
    if ( m_linac.A_leaf_aabb[ index ].xmin < m_linac.A_bank_aabb.xmin )
    {
        m_linac.A_bank_aabb.xmin = m_linac.A_leaf_aabb[ index ].xmin;
    }

    if ( m_linac.A_leaf_aabb[ index ].ymin < m_linac.A_bank_aabb.ymin )
    {
        m_linac.A_bank_aabb.ymin = m_linac.A_leaf_aabb[ index ].ymin;
    }

    if ( m_linac.A_leaf_aabb[ index ].zmin < m_linac.A_bank_aabb.zmin )
    {
        m_linac.A_bank_aabb.zmin = m_linac.A_leaf_aabb[ index ].zmin;
    }

    if ( m_linac.A_leaf_aabb[ index ].xmax > m_linac.A_bank_aabb.xmax )
    {
        m_linac.A_bank_aabb.xmax = m_linac.A_leaf_aabb[ index ].xmax;
    }

    if ( m_linac.A_leaf_aabb[ index ].ymax > m_linac.A_bank_aabb.ymax )
    {
        m_linac.A_bank_aabb.ymax = m_linac.A_leaf_aabb[ index ].ymax;
    }

    if ( m_linac.A_leaf_aabb[ index ].zmax > m_linac.A_bank_aabb.zmax )
    {
        m_linac.A_bank_aabb.zmax = m_linac.A_leaf_aabb[ index ].zmax;
    }

}

void MeshPhanLINACNav::m_translate_leaf_B( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.B_leaf_index[ index ];
    ui32 nb_tri = m_linac.B_leaf_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.B_leaf_v1[ offset + i_tri ] = fxyz_add( m_linac.B_leaf_v1[ offset + i_tri ], T );
        m_linac.B_leaf_v2[ offset + i_tri ] = fxyz_add( m_linac.B_leaf_v2[ offset + i_tri ], T );
        m_linac.B_leaf_v3[ offset + i_tri ] = fxyz_add( m_linac.B_leaf_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.B_leaf_aabb[ index ].xmin += T.x;
    m_linac.B_leaf_aabb[ index ].xmax += T.x;
    m_linac.B_leaf_aabb[ index ].ymin += T.y;
    m_linac.B_leaf_aabb[ index ].ymax += T.y;
    m_linac.B_leaf_aabb[ index ].zmin += T.z;
    m_linac.B_leaf_aabb[ index ].zmax += T.z;

    // Update the bank AABB
    if ( m_linac.B_leaf_aabb[ index ].xmin < m_linac.B_bank_aabb.xmin )
    {
        m_linac.B_bank_aabb.xmin = m_linac.B_leaf_aabb[ index ].xmin;
    }

    if ( m_linac.B_leaf_aabb[ index ].ymin < m_linac.B_bank_aabb.ymin )
    {
        m_linac.B_bank_aabb.ymin = m_linac.B_leaf_aabb[ index ].ymin;
    }

    if ( m_linac.B_leaf_aabb[ index ].zmin < m_linac.B_bank_aabb.zmin )
    {
        m_linac.B_bank_aabb.zmin = m_linac.B_leaf_aabb[ index ].zmin;
    }

    if ( m_linac.B_leaf_aabb[ index ].xmax > m_linac.B_bank_aabb.xmax )
    {
        m_linac.B_bank_aabb.xmax = m_linac.B_leaf_aabb[ index ].xmax;
    }

    if ( m_linac.B_leaf_aabb[ index ].ymax > m_linac.B_bank_aabb.ymax )
    {
        m_linac.B_bank_aabb.ymax = m_linac.B_leaf_aabb[ index ].ymax;
    }

    if ( m_linac.B_leaf_aabb[ index ].zmax > m_linac.B_bank_aabb.zmax )
    {
        m_linac.B_bank_aabb.zmax = m_linac.B_leaf_aabb[ index ].zmax;
    }

}

void MeshPhanLINACNav::m_configure_linac()
{

    // Open the beam file
    std::ifstream file( m_beam_config_filename.c_str(), std::ios::in );
    if( !file )
    {
        GGcerr << "Error to open the Beam file'" << m_beam_config_filename << "'!" << GGendl;
        exit_simulation();
    }

    std::string line;
    std::vector< std::string > keys;

    // Look for the beam number
    bool find_beam = false;
    while ( file )
    {
        // Read a line
        std::getline( file, line );
        keys = m_split_txt( line );

        if ( keys.size() >= 3 )
        {
            if ( keys[ 0 ] == "Beam" && std::stoi( keys[ 2 ] ) == m_beam_index )
            {
                find_beam = true;
                break;
            }
        }
    }

    if ( !find_beam )
    {
        GGcerr << "Beam configuration error: beam " << m_beam_index << " was not found!" << GGendl;
        exit_simulation();
    }

//    GGcout << "Find beam: " << line << GGendl;

    // Then look for the number of fields
    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find("Number of Fields") != std::string::npos )
        {
            break;
        }
    }

//    GGcout << "Find nb field: " << line << GGendl;

    keys = m_split_txt( line );
    ui32 nb_fields = std::stoi( keys[ 4 ] );

    if ( m_field_index >= nb_fields )
    {
        GGcerr << "Out of index for the field number, asked: " << m_field_index
               << " but a total of field of " << nb_fields << GGendl;
        exit_simulation();
    }    

    // Look for the number of leaves
    bool find_field = false;
    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find("Number of Leaves") != std::string::npos )
        {
            find_field = true;
            break;
        }
    }

    if ( !find_field )
    {
        GGcerr << "Beam configuration error: field " << m_field_index << " was not found!" << GGendl;
        exit_simulation();
    }

//    GGcout << "Find nb leaves: " << line << GGendl;

    keys = m_split_txt( line );
    ui32 nb_leaves = std::stoi( keys[ 4 ] );
    if ( m_linac.A_nb_leaves + m_linac.B_nb_leaves != nb_leaves )
    {
        GGcerr << "Beam configuration error, " << nb_leaves
               << " leaves were found but LINAC model have " << m_linac.A_nb_leaves + m_linac.B_nb_leaves
               << " leaves!" << GGendl;
        exit_simulation();
    }

    // Search the required field
    while ( file )
    {
        // Read a line
        std::getline( file, line );
        keys = m_split_txt( line );

        if ( keys.size() >= 3 )
        {
            if ( keys[ 0 ] == "Control" && std::stoi( keys[ 2 ] ) == m_field_index )
            {
                break;
            }
        }
    }

//    GGcout << "Find field: " << line << GGendl;

    // Then read the index CDF (not use at the time, so skip the line)
    std::getline( file, line );

    // Get the gantry angle
    std::getline( file, line );

    // Check
    if ( line.find( "Gantry Angle" ) == std::string::npos )
    {
        GGcerr << "Beam configuration error, no gantry angle was found!" << GGendl;
        exit_simulation();
    }

    // Read gantry angle values
    keys = m_split_txt( line );

    // if only one angle, rotate around the z-axis
    if ( keys.size() == 4 )
    {
        m_rot_linac = make_f32xyz( 0.0, 0.0, std::stof( keys[ 3 ] ) *deg );
    }
    else if ( keys.size() == 6 ) // non-coplanar beam, or rotation on the carousel
    {
        m_rot_linac = make_f32xyz( std::stof( keys[ 3 ] ) *deg,
                                   std::stof( keys[ 4 ] ) *deg,
                                   std::stof( keys[ 5 ] ) *deg );
    }
    else // otherwise, it seems that there is an error somewhere
    {
        GGcerr << "Beam configuration error, gantry angle must have one angle or the three rotation angles: "
               << keys.size() - 3 << " angles found!" << GGendl;
        exit_simulation();
    }

    // Get the transformation matrix to map local to global coordinate
    TransformCalculator *trans = new TransformCalculator;
    trans->set_translation( m_pos_mlc );
    trans->set_rotation( m_rot_linac );
    trans->set_axis_transformation( m_axis_linac );
    m_linac.transform = trans->get_transformation_matrix();
    delete trans;

    //// JAWS //////////////////////////////////////////

    // Next four lines should the jaw config
    f32 jaw_x_min = 0.0; bool jaw_x = false;
    f32 jaw_x_max = 0.0;
    f32 jaw_y_min = 0.0; bool jaw_y = false;
    f32 jaw_y_max = 0.0;

    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find( "Jaw" ) != std::string::npos )
        {
            keys = m_split_txt( line );
            if ( keys[ 1 ] == "X" && keys[ 2 ] == "min" )
            {
                jaw_x_min = std::stof( keys[ 4 ] );
                jaw_x = true;
            }
            if ( keys[ 1 ] == "X" && keys[ 2 ] == "max" )
            {
                jaw_x_max = std::stof( keys[ 4 ] );
                jaw_x = true;
            }
            if ( keys[ 1 ] == "Y" && keys[ 2 ] == "min" )
            {
                jaw_y_min = std::stof( keys[ 4 ] );
                jaw_y = true;
            }
            if ( keys[ 1 ] == "Y" && keys[ 2 ] == "max" )
            {
                jaw_y_max = std::stof( keys[ 4 ] );
                jaw_y = true;
            }
        }
        else
        {
            break;
        }
    }

    // Check
    if ( !jaw_x && m_linac.X_nb_jaw != 0 )
    {
        GGcerr << "Beam configuration error, geometry of the jaw-X was defined but the position values were not found!" << GGendl;
        exit_simulation();
    }
    if ( !jaw_y && m_linac.Y_nb_jaw != 0 )
    {
        GGcerr << "Beam configuration error, geometry of the jaw-Y was defined but the position values were not found!" << GGendl;
        exit_simulation();
    }

    // Configure the jaws
    if ( m_linac.X_nb_jaw != 0 )
    {
        m_translate_jaw_x( 0, make_f32xyz( jaw_x_max, 0.0, 0.0 ) );   // X1 ( x > 0 )
        m_translate_jaw_x( 1, make_f32xyz( jaw_x_min, 0.0, 0.0 ) );   // X2 ( x < 0 )
    }

    if ( m_linac.Y_nb_jaw != 0 )
    {
        m_translate_jaw_y( 0, make_f32xyz( 0.0, jaw_y_max, 0.0 ) );   // Y1 ( y > 0 )
        m_translate_jaw_y( 1, make_f32xyz( 0.0, jaw_y_min, 0.0 ) );   // Y2 ( y < 0 )
    }

    //// LEAVES BANK A ///////////////////////////////////////////////

    ui32 ileaf = 0;
    bool wd_leaf = false; // watchdog
    while ( file )
    {
        if ( line.find( "Leaf" ) != std::string::npos && line.find( "A" ) != std::string::npos )
        {
            // If first leaf of the bank A, check
            if ( ileaf == 0 )
            {
                keys = m_split_txt( line );
                if ( keys[ 1 ] != "1A" )
                {
                    GGcerr << "Beam configuration error, first leaf of the bank A must start by index '1A': " << keys[ 1 ]
                           << " found." << GGendl;
                    exit_simulation();
                }
            }

            // watchdog
            if ( ileaf >= m_linac.A_nb_leaves )
            {
                GGcerr << "Beam configuration error, find more leaves in the configuration "
                       << "file for the bank A than leaves in the LINAC model!" << GGendl;
                exit_simulation();
            }

            // find at least one leaf
            if ( !wd_leaf ) wd_leaf = true;

            // read data and move the leaf
            keys = m_split_txt( line );
            m_translate_leaf_A( ileaf++, make_f32xyz( std::stof( keys[ 3 ] ), 0.0, 0.0 ) );

        }
        else
        {
            break;
        }

        // Read a line
        std::getline( file, line );
    }

    // No leaves were found
    if ( !wd_leaf )
    {
        GGcerr << "Beam configuration error, no leaves from the bank A were found!" << GGendl;
        exit_simulation();
    }

    //// LEAVES BANK B ///////////////////////////////////////////////

    ileaf = 0;
    wd_leaf = false; // watchdog
    while ( file )
    {

        if ( line.find( "Leaf" ) != std::string::npos && line.find( "B" ) != std::string::npos )
        {
            // If first leaf of the bank A, check
            if ( ileaf == 0 )
            {
                keys = m_split_txt( line );
                if ( keys[ 1 ] != "1B" )
                {
                    GGcerr << "Beam configuration error, first leaf of the bank B must start by index '1B': " << keys[ 1 ]
                           << " found." << GGendl;
                    exit_simulation();
                }
            }

            // watchdog
            if ( ileaf >= m_linac.B_nb_leaves )
            {
                GGcerr << "Beam configuration error, find more leaves in the configuration "
                       << "file for the bank B than leaves in the LINAC model!" << GGendl;
                exit_simulation();
            }

            // find at least one leaf
            if ( !wd_leaf ) wd_leaf = true;

            // read data and move the leaf
            keys = m_split_txt( line );
            m_translate_leaf_B( ileaf++, make_f32xyz( std::stof( keys[ 3 ] ), 0.0, 0.0 ) );

        }
        else
        {
            break;
        }

        // Read a line
        std::getline( file, line );
    }

    // No leaves were found
    if ( !wd_leaf )
    {
        GGcerr << "Beam configuration error, no leaves from the bank B were found!" << GGendl;
        exit_simulation();
    }

    // Finally compute the global bounding box of the LINAC
    f32 xmin = FLT_MAX; f32 xmax = -FLT_MAX;
    f32 ymin = FLT_MAX; f32 ymax = -FLT_MAX;
    f32 zmin = FLT_MAX; f32 zmax = -FLT_MAX;

    if ( m_linac.A_bank_aabb.xmin < xmin ) xmin = m_linac.A_bank_aabb.xmin;
    if ( m_linac.B_bank_aabb.xmin < xmin ) xmin = m_linac.B_bank_aabb.xmin;
    if ( m_linac.A_bank_aabb.ymin < ymin ) ymin = m_linac.A_bank_aabb.ymin;
    if ( m_linac.B_bank_aabb.ymin < ymin ) ymin = m_linac.B_bank_aabb.ymin;
    if ( m_linac.A_bank_aabb.zmin < zmin ) zmin = m_linac.A_bank_aabb.zmin;
    if ( m_linac.B_bank_aabb.zmin < zmin ) zmin = m_linac.B_bank_aabb.zmin;

    if ( m_linac.A_bank_aabb.xmax > xmax ) xmax = m_linac.A_bank_aabb.xmax;
    if ( m_linac.B_bank_aabb.xmax > xmax ) xmax = m_linac.B_bank_aabb.xmax;
    if ( m_linac.A_bank_aabb.ymax > ymax ) ymax = m_linac.A_bank_aabb.ymax;
    if ( m_linac.B_bank_aabb.ymax > ymax ) ymax = m_linac.B_bank_aabb.ymax;
    if ( m_linac.A_bank_aabb.zmax > zmax ) zmax = m_linac.A_bank_aabb.zmax;
    if ( m_linac.B_bank_aabb.zmax > zmax ) zmax = m_linac.B_bank_aabb.zmax;

    if ( m_linac.X_nb_jaw != 0 )
    {
        if ( m_linac.X_jaw_aabb[ 0 ].xmin < xmin ) xmin = m_linac.X_jaw_aabb[ 0 ].xmin;
        if ( m_linac.X_jaw_aabb[ 1 ].xmin < xmin ) xmin = m_linac.X_jaw_aabb[ 1 ].xmin;
        if ( m_linac.X_jaw_aabb[ 0 ].ymin < ymin ) ymin = m_linac.X_jaw_aabb[ 0 ].ymin;
        if ( m_linac.X_jaw_aabb[ 1 ].ymin < ymin ) ymin = m_linac.X_jaw_aabb[ 1 ].ymin;
        if ( m_linac.X_jaw_aabb[ 0 ].zmin < zmin ) zmin = m_linac.X_jaw_aabb[ 0 ].zmin;
        if ( m_linac.X_jaw_aabb[ 1 ].zmin < zmin ) zmin = m_linac.X_jaw_aabb[ 1 ].zmin;

        if ( m_linac.X_jaw_aabb[ 0 ].xmax > xmax ) xmax = m_linac.X_jaw_aabb[ 0 ].xmax;
        if ( m_linac.X_jaw_aabb[ 1 ].xmax > xmax ) xmax = m_linac.X_jaw_aabb[ 1 ].xmax;
        if ( m_linac.X_jaw_aabb[ 0 ].ymax > ymax ) ymax = m_linac.X_jaw_aabb[ 0 ].ymax;
        if ( m_linac.X_jaw_aabb[ 1 ].ymax > ymax ) ymax = m_linac.X_jaw_aabb[ 1 ].ymax;
        if ( m_linac.X_jaw_aabb[ 0 ].zmax > zmax ) zmax = m_linac.X_jaw_aabb[ 0 ].zmax;
        if ( m_linac.X_jaw_aabb[ 1 ].zmax > zmax ) zmax = m_linac.X_jaw_aabb[ 1 ].zmax;
    }

    if ( m_linac.Y_nb_jaw != 0 )
    {
        if ( m_linac.Y_jaw_aabb[ 0 ].xmin < xmin ) xmin = m_linac.Y_jaw_aabb[ 0 ].xmin;
        if ( m_linac.Y_jaw_aabb[ 1 ].xmin < xmin ) xmin = m_linac.Y_jaw_aabb[ 1 ].xmin;
        if ( m_linac.Y_jaw_aabb[ 0 ].ymin < ymin ) ymin = m_linac.Y_jaw_aabb[ 0 ].ymin;
        if ( m_linac.Y_jaw_aabb[ 1 ].ymin < ymin ) ymin = m_linac.Y_jaw_aabb[ 1 ].ymin;
        if ( m_linac.Y_jaw_aabb[ 0 ].zmin < zmin ) zmin = m_linac.Y_jaw_aabb[ 0 ].zmin;
        if ( m_linac.Y_jaw_aabb[ 1 ].zmin < zmin ) zmin = m_linac.Y_jaw_aabb[ 1 ].zmin;

        if ( m_linac.Y_jaw_aabb[ 0 ].xmax > xmax ) xmax = m_linac.Y_jaw_aabb[ 0 ].xmax;
        if ( m_linac.Y_jaw_aabb[ 1 ].xmax > xmax ) xmax = m_linac.Y_jaw_aabb[ 1 ].xmax;
        if ( m_linac.Y_jaw_aabb[ 0 ].ymax > ymax ) ymax = m_linac.Y_jaw_aabb[ 0 ].ymax;
        if ( m_linac.Y_jaw_aabb[ 1 ].ymax > ymax ) ymax = m_linac.Y_jaw_aabb[ 1 ].ymax;
        if ( m_linac.Y_jaw_aabb[ 0 ].zmax > zmax ) zmax = m_linac.Y_jaw_aabb[ 0 ].zmax;
        if ( m_linac.Y_jaw_aabb[ 1 ].zmax > zmax ) zmax = m_linac.Y_jaw_aabb[ 1 ].zmax;
    }

    // Store the data
    m_linac.aabb.xmin = xmin;
    m_linac.aabb.xmax = xmax;
    m_linac.aabb.ymin = ymin;
    m_linac.aabb.ymax = ymax;
    m_linac.aabb.zmin = zmin;
    m_linac.aabb.zmax = zmax;

}

// return memory usage
ui64 MeshPhanLINACNav::m_get_memory_usage()
{
    ui64 mem = 0;

    // Get tot nb triangles
    ui32 tot_tri = 0;

    ui32 ileaf = 0; while ( ileaf < m_linac.A_nb_leaves )
    {
        tot_tri += m_linac.A_leaf_nb_triangles[ ileaf++ ];
    }

    ileaf = 0; while ( ileaf < m_linac.B_nb_leaves )
    {
        tot_tri += m_linac.B_leaf_nb_triangles[ ileaf++ ];
    }

    if ( m_linac.X_nb_jaw != 0 )
    {
        tot_tri += m_linac.X_jaw_nb_triangles[ 0 ];
        tot_tri += m_linac.X_jaw_nb_triangles[ 1 ];
    }

    if ( m_linac.Y_nb_jaw != 0 )
    {
        tot_tri += m_linac.Y_jaw_nb_triangles[ 0 ];
        tot_tri += m_linac.Y_jaw_nb_triangles[ 1 ];
    }

    // All tri
    mem = 3 * tot_tri * sizeof( f32xyz );

    // Bank A
    mem += m_linac.A_nb_leaves * 2 * sizeof( ui32 ); // index, nb tri
    mem += m_linac.A_nb_leaves * 6 * sizeof( f32 );  // aabb
    mem += 6 * sizeof( f32 ) + sizeof( ui32 );       // main aabb, nb leaves

    // Bank B
    mem += m_linac.B_nb_leaves * 2 * sizeof( ui32 ); // index, nb tri
    mem += m_linac.B_nb_leaves * 6 * sizeof( f32 );  // aabb
    mem += 6 * sizeof( f32 ) + sizeof( ui32 );       // main aabb, nb leaves

    // Jaws X
    mem += m_linac.X_nb_jaw * 2 * sizeof( ui32 );    // inedx, nb tri
    mem += 6 * sizeof( f32 ) + sizeof( ui32 );       // main aabb, nb jaws

    // Jaws Y
    mem += m_linac.Y_nb_jaw * 2 * sizeof( ui32 );    // inedx, nb tri
    mem += 6 * sizeof( f32 ) + sizeof( ui32 );       // main aabb, nb jaws

    // Global aabb
    mem += 6 * sizeof( f32 );

    return mem;
}

//// Setting/Getting functions

void MeshPhanLINACNav::set_mlc_meshes( std::string filename )
{
    m_mlc_filename = filename;
}

void MeshPhanLINACNav::set_jaw_x_meshes( std::string filename )
{
    m_jaw_x_filename = filename;
}

void MeshPhanLINACNav::set_jaw_y_meshes( std::string filename )
{
    m_jaw_y_filename = filename;
}

void MeshPhanLINACNav::set_beam_configuration( std::string filename, ui32 beam_index, ui32 field_index )
{
    m_beam_config_filename = filename;
    m_beam_index = beam_index;
    m_field_index = field_index;
}

void MeshPhanLINACNav::set_number_of_leaves( ui32 nb_bank_A, ui32 nb_bank_B )
{
    m_linac.A_nb_leaves = nb_bank_A;
    m_linac.B_nb_leaves = nb_bank_B;
}

void MeshPhanLINACNav::set_mlc_position( f32 px, f32 py, f32 pz )
{
    m_pos_mlc = make_f32xyz( px, py, pz );
}

void MeshPhanLINACNav::set_local_jaw_x_position( f32 px, f32 py, f32 pz )
{
    m_loc_pos_jaw_x = make_f32xyz( px, py, pz );
}

void MeshPhanLINACNav::set_local_jaw_y_position( f32 px, f32 py, f32 pz )
{
    m_loc_pos_jaw_y = make_f32xyz( px, py, pz );
}

void MeshPhanLINACNav::set_linac_local_axis( f32 m00, f32 m01, f32 m02,
                                             f32 m10, f32 m11, f32 m12,
                                             f32 m20, f32 m21, f32 m22 )
{
    m_axis_linac = make_f32matrix33( m00, m01, m02,
                                     m10, m11, m12,
                                     m20, m21, m22 );
}

void MeshPhanLINACNav::set_navigation_within_mlc( bool flag )
{
    m_nav_within_mlc = flag;
}

void MeshPhanLINACNav::set_linac_material( std::string mat_name )
{
    m_linac_material[ 0 ] = mat_name;
}

LinacData MeshPhanLINACNav::get_linac_geometry()
{
    return m_linac;
}

f32matrix44 MeshPhanLINACNav::get_linac_transformation()
{
    return m_linac.transform;
}

void MeshPhanLINACNav::set_materials(std::string filename )
{
    m_materials_filename = filename;
}

////// Main functions

MeshPhanLINACNav::MeshPhanLINACNav ()
{
    // Leaves in Bank A
    m_linac.A_leaf_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.A_leaf_v2 = NULL;           // Vertex 2
    m_linac.A_leaf_v3 = NULL;           // Vertex 3
    m_linac.A_leaf_index = NULL;        // Index to acces to a leaf
    m_linac.A_leaf_nb_triangles = NULL; // Nb of triangles within each leaf
    m_linac.A_leaf_aabb = NULL;         // Bounding box of each leaf

    m_linac.A_bank_aabb.xmin = 0.0;     // Bounding box of the bank A
    m_linac.A_bank_aabb.xmax = 0.0;
    m_linac.A_bank_aabb.ymin = 0.0;
    m_linac.A_bank_aabb.ymax = 0.0;
    m_linac.A_bank_aabb.zmin = 0.0;
    m_linac.A_bank_aabb.zmax = 0.0;

    m_linac.A_nb_leaves = 0;            // Number of leaves in the bank A

    // Leaves in Bank B
    m_linac.B_leaf_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.B_leaf_v2 = NULL;           // Vertex 2
    m_linac.B_leaf_v3 = NULL;           // Vertex 3
    m_linac.B_leaf_index = NULL;        // Index to acces to a leaf
    m_linac.B_leaf_nb_triangles = NULL; // Nb of triangles within each leaf
    m_linac.B_leaf_aabb = NULL;         // Bounding box of each leaf

    m_linac.B_bank_aabb.xmin = 0.0;     // Bounding box of the bank B
    m_linac.B_bank_aabb.xmax = 0.0;
    m_linac.B_bank_aabb.ymin = 0.0;
    m_linac.B_bank_aabb.ymax = 0.0;
    m_linac.B_bank_aabb.zmin = 0.0;
    m_linac.B_bank_aabb.zmax = 0.0;

    m_linac.B_nb_leaves = 0;            // Number of leaves in the bank B

    // Jaws X
    m_linac.X_jaw_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.X_jaw_v2 = NULL;           // Vertex 2
    m_linac.X_jaw_v3 = NULL;           // Vertex 3
    m_linac.X_jaw_index = NULL;        // Index to acces to a jaw
    m_linac.X_jaw_nb_triangles = NULL; // Nb of triangles within each jaw
    m_linac.X_jaw_aabb = NULL;         // Bounding box of each jaw
    m_linac.X_nb_jaw = 0;              // Number of jaws

    // Jaws Y
    m_linac.Y_jaw_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.Y_jaw_v2 = NULL;           // Vertex 2
    m_linac.Y_jaw_v3 = NULL;           // Vertex 3
    m_linac.Y_jaw_index = NULL;        // Index to acces to a jaw
    m_linac.Y_jaw_nb_triangles = NULL; // Nb of triangles within each jaw
    m_linac.Y_jaw_aabb = NULL;         // Bounding box of each jaw
    m_linac.Y_nb_jaw = 0;              // Number of jaws

    m_linac.aabb.xmin = 0.0;           // Bounding box of the LINAC
    m_linac.aabb.xmax = 0.0;
    m_linac.aabb.ymin = 0.0;
    m_linac.aabb.ymax = 0.0;
    m_linac.aabb.zmin = 0.0;
    m_linac.aabb.zmax = 0.0;

    m_linac.transform = make_f32matrix44_zeros();

    set_name( "MeshPhanLINACNav" );
    m_mlc_filename = "";
    m_jaw_x_filename = "";
    m_jaw_y_filename = "";
    m_beam_config_filename = "";

    m_pos_mlc = make_f32xyz_zeros();
    m_loc_pos_jaw_x = make_f32xyz_zeros();
    m_loc_pos_jaw_y = make_f32xyz_zeros();
    m_rot_linac = make_f32xyz_zeros();
    m_axis_linac = make_f32matrix33_zeros();

    m_beam_index = 0;
    m_field_index = 0;

    m_nav_within_mlc = true;
    m_materials_filename = "";
    m_linac_material.push_back("");

}

//// Mandatory functions

void MeshPhanLINACNav::track_to_in( Particles particles )
{
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id=0;
        while ( id<particles.size )
        {
            MPLINACN::kernel_host_track_to_in( particles.data_h, m_linac,
                                               m_params.data_h.geom_tolerance,
                                               id );
            ++id;
        }
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

        MPLINACN::kernel_device_track_to_in<<<grid, threads>>> ( particles.data_d, m_linac,
                                                                 m_params.data_d.geom_tolerance );
        cuda_error_check ( "Error ", " Kernel_MeshPhanLINACNav (track to in)" );
        cudaThreadSynchronize();
    }
}

void MeshPhanLINACNav::track_to_out( Particles particles )
{


    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id=0;
        while ( id<particles.size )
        {
            MPLINACN::kernel_host_track_to_out( particles.data_h, m_linac,
                                                m_materials.data_h, m_cross_sections.photon_CS.data_h,
                                                m_params.data_h,
                                                m_nav_within_mlc,
                                                id );
            ++id;
        }
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

        MPLINACN::kernel_device_track_to_out<<<grid, threads>>> ( particles.data_d, m_linac,
                                                                  m_materials.data_d, m_cross_sections.photon_CS.data_d,
                                                                  m_params.data_d, m_nav_within_mlc );
        cuda_error_check ( "Error ", " Kernel_MeshPhanLINACNav (track to in)" );
        cudaThreadSynchronize();
    }

}

void MeshPhanLINACNav::initialize( GlobalSimulationParameters params )
{
    // Check params
    if ( m_mlc_filename == "" )
    {
        GGcerr << "No mesh file specified for MLC of the LINAC phantom!" << GGendl;
        exit_simulation();
    }

    if ( m_linac.A_nb_leaves == 0 && m_linac.B_nb_leaves == 0 )
    {
        GGcerr << "MeshPhanLINACNav: number of leaves per bank must be specified!" << GGendl;
        exit_simulation();
    }

    if ( m_nav_within_mlc && ( m_materials_filename == "" || m_linac_material[ 0 ] == "" ) )
    {
        GGcerr << "MeshPhanLINACNav: navigation required but material information was not provided!" << GGendl;
        exit_simulation();
    }

    // Params
    m_params = params;

    // Init MLC
    m_init_mlc();    

    // If jaw x is defined, init
    if ( m_jaw_x_filename != "" )
    {
        m_init_jaw_x();

        // move the jaw relatively to the mlc (local frame)
        m_translate_jaw_x( 0, m_loc_pos_jaw_x );
        m_translate_jaw_x( 1, m_loc_pos_jaw_x );

    }

    // If jaw y is defined, init
    if ( m_jaw_y_filename != "" )
    {
        m_init_jaw_y();

        // move the jaw relatively to the mlc (local frame)
        m_translate_jaw_y( 0, m_loc_pos_jaw_y );
        m_translate_jaw_y( 1, m_loc_pos_jaw_y );
    }

    // Configure the linac
    m_configure_linac();

    // Init materials
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize( m_linac_material, params );

    // Cross Sections
    m_cross_sections.initialize( m_materials, params );

    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem( "MeshPhanLINACNav", mem );
    }

    /*

    // Materials table
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize( m_phantom.list_of_materials, params );    

    // Cross Sections
    m_cross_sections.initialize( m_materials, params );   


    */
}




#endif
