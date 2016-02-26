// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_img_nav.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef VOX_PHAN_DOSI_NAV_CU
#define VOX_PHAN_DOSI_NAV_CU

#include "vox_phan_dosi_nav.cuh"

////:: GPU Codes

__host__ __device__ void VPDN::track_electron_to_out ( ParticlesData &particles,
                                                       VoxVolumeData vol,
                                                       MaterialsTable materials,
                                                       ElectronsCrossSectionTable electron_CS_table,
                                                       GlobalSimulationParametersData parameters,
                                                       DoseData &dosi,
                                                       f32 &randomnumbereIoni,
                                                       f32 &randomnumbereBrem,
                                                       f32 freeLength,
                                                       ui32 part_id )
{

    // Parameters values need to be stored for every e-step
    f32 alongStepLength = 0.;               // Distance from the last physics interaction.
    bool lastStepisaPhysicEffect = TRUE;    // To store last random number
    bool bool_loop = true;                  // If it is not the last step in the same voxel
    //bool secondaryParticleCreated = FALSE;  // If a secondary particle is created

    alongStepLength = freeLength;
    if ( freeLength>0.0 ) lastStepisaPhysicEffect = FALSE; // Changement de voxel sans effet physique

    // Parameters
    f32 trueStepLength = FLT_MAX;
    f32 totalLength = 0.;
    f32 par1, par2;
    f32xyz pos, dir;    // particle state
    f32 energy;

    // Some other params
    f32 lengthtoVertex;               // Value to store the distance from the last physics interaction.
    ui8 next_discrete_process ;
    ui32 table_index;                 // indice de lecture de table de sections efficaces
    f32 next_interaction_distance = FLT_MAX;
    f32 dedxeIoni = 0;
    f32 dedxeBrem = 0;
    f32 erange = 0;
    f32 lambda = 0;
    bool significant_loss;
    f32 edep;
    f32 trueGeomLength;

    f32 electronEcut;

#ifdef DEBUG
    // DEBUG
    ui32 istep = 0;
#endif

    do
    {
        // Init step
        next_interaction_distance = FLT_MAX;
        dedxeIoni = 0;
        dedxeBrem = 0;
        erange = 0;
        lambda = 0;

        // Stop simulation if out of the phantom
        if ( !test_point_AABB_with_tolerance ( make_f32xyz( particles.px[ part_id ], particles.py[ part_id ], particles.pz[ part_id ] ),
                                               vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax, parameters.geom_tolerance ) )
        {
            particles.endsimu[ part_id ] = PARTICLE_FREEZE;
            return;
        }

        // Get Random number stored until a physic interaction
        if ( lastStepisaPhysicEffect == TRUE )
        {
            randomnumbereBrem = -logf ( prng_uniform( &(particles.prng[part_id]) ) );
            randomnumbereIoni = -logf ( prng_uniform( &(particles.prng[part_id]) ) );
            alongStepLength = 0.f;
            lastStepisaPhysicEffect = FALSE;
        }

        // Read position
        pos.x = particles.px[part_id];
        pos.y = particles.py[part_id];
        pos.z = particles.pz[part_id];

        // Read direction
        dir.x = particles.dx[part_id];
        dir.y = particles.dy[part_id];
        dir.z = particles.dz[part_id];

        // Read energy
        energy = particles.E[part_id];

        // Defined index phantom
        f32xyz ivoxsize;
        ivoxsize.x = 1.0 / vol.spacing_x;
        ivoxsize.y = 1.0 / vol.spacing_y;
        ivoxsize.z = 1.0 / vol.spacing_z;
        ui32xyzw index_phantom;
        index_phantom.x = ui32 ( ( pos.x + vol.off_x ) * ivoxsize.x );
        index_phantom.y = ui32 ( ( pos.y + vol.off_y ) * ivoxsize.y );
        index_phantom.z = ui32 ( ( pos.z + vol.off_z ) * ivoxsize.z );

        index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                + index_phantom.y*vol.nb_vox_x
                + index_phantom.x; // linear index

#ifdef DEBUG
        assert( index_phantom.w < vol.number_of_voxels );
#endif

        // Get the material that compose this volume
        ui16 mat_id = vol.values[ index_phantom.w ];
        electronEcut = materials.electron_energy_cut[ mat_id ];

#ifdef DEBUG
        assert( mat_id < 65536 );
#endif

        // Read the different CS, dE/dx tables
        e_read_CS_table ( mat_id, energy, electron_CS_table, next_discrete_process, table_index,
                          next_interaction_distance, dedxeIoni, dedxeBrem, erange, lambda, randomnumbereBrem, randomnumbereIoni, parameters );

        //printf("   PhysSteplength %f\n", next_interaction_distance);

        // Vertex length
        lengthtoVertex = ( alongStepLength > next_interaction_distance ) ? 0. : next_interaction_distance - alongStepLength;

        //Get cut step
        f32 cutstep = StepFunction ( erange );

        if ( lengthtoVertex > cutstep )
        {
            significant_loss = true;
            trueStepLength = cutstep;
        }
        else
        {
            significant_loss = false;
            trueStepLength = lengthtoVertex;
        }

        //// Get the next distance boundary volume /////////////////////////////////

        // get voxel params
        f32 vox_xmin = index_phantom.x*vol.spacing_x - vol.off_x;
        f32 vox_ymin = index_phantom.y*vol.spacing_y - vol.off_y;
        f32 vox_zmin = index_phantom.z*vol.spacing_z - vol.off_z;
        f32 vox_xmax = vox_xmin + vol.spacing_x;
        f32 vox_ymax = vox_ymin + vol.spacing_y;
        f32 vox_zmax = vox_zmin + vol.spacing_z;

        // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
        // TODO: In theory this have to be applied just at the entry of the particle within the volume
        //       in order to avoid particle entry between voxels. Then, computing improvement can be made
        //       by calling this function only once, just for the particle step=0.    - JB
        pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                                vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters.geom_tolerance );

        // compute the next distance boundary
        f32 boundary_distance = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                               vox_ymin, vox_ymax, vox_zmin, vox_zmax );

#ifdef DEBUG
        assert( boundary_distance != FLT_MAX );
#endif

        if ( trueStepLength > boundary_distance )
        {
            if ( parameters.physics_list[ELECTRON_MSC] == ENABLED )
            {
                trueGeomLength = gTransformToGeom ( trueStepLength, erange, lambda, energy,
                                                    par1, par2, electron_CS_table, mat_id );

                if ( trueGeomLength > boundary_distance )
                {
                    bool_loop=false;
                }
            }
            else
            {
                bool_loop = false;
            }
        }

        if ( bool_loop==true )
        {
            
            if ( significant_loss == true )
            {

                // Energy loss (call eFluctuation)
                edep = eLoss ( trueStepLength, particles.E[ part_id ], dedxeIoni, dedxeBrem, erange,
                               electron_CS_table, mat_id, materials, particles, parameters, part_id );

                GlobalMscScattering ( trueStepLength, cutstep, erange, energy, lambda, dedxeIoni,
                                      dedxeBrem,  electron_CS_table,  mat_id, particles, part_id, par1, par2,    // HERE particle move
                                      materials, dosi, index_phantom, vol, parameters );
                
                dose_record_standard ( dosi, edep, particles.px[ part_id ], particles.py[ part_id ], particles.pz[ part_id ] );

                alongStepLength += trueStepLength;
                totalLength += trueStepLength;
                lastStepisaPhysicEffect = FALSE;

            }
            else
            {

                //// InvokeAlongStepDoItProcs ////////////////////////////////////////////////////////////////////////

                // Energy loss (call eFluctuation)
                edep = eLoss ( trueStepLength, particles.E[ part_id ], dedxeIoni, dedxeBrem, erange,
                               electron_CS_table, mat_id, materials, particles, parameters, part_id );

                GlobalMscScattering ( trueStepLength, lengthtoVertex, erange, energy, lambda,   dedxeIoni,
                                      dedxeBrem,   electron_CS_table,  mat_id, particles,  part_id, par1, par2,     // HERE particle move
                                      materials, dosi, index_phantom, vol, parameters );

                dose_record_standard ( dosi, edep, particles.px[part_id], particles.py[part_id], particles.pz[part_id] );

                //// InvokePostStepDoItProcs ////////////////////////////////////////////////////////////////////////


                SecParticle secondary_part;
                secondary_part.E = 0.;
                secondary_part.endsimu = PARTICLE_DEAD;

                if ( next_discrete_process == ELECTRON_IONISATION )
                {

                    secondary_part = eSampleSecondarieElectron ( electronEcut, particles,  part_id );
                    lastStepisaPhysicEffect = TRUE;

                }
                else if ( next_discrete_process == ELECTRON_BREMSSTRAHLUNG )
                {
                    /// TODO return a photon - JB
                    eSampleSecondarieGamma ( parameters.cs_table_min_E, parameters.cs_table_max_E, particles, part_id, materials, mat_id );
                    lastStepisaPhysicEffect = TRUE;
                }

                /// If there is a secondary particle, push the primary into buffer and track this new particle

                /// Handle secondary //////////////////////

                if ( secondary_part.endsimu == PARTICLE_ALIVE &&
                     particles.level[ part_id ] < parameters.nb_of_secondaries && parameters.secondaries_list[ELECTRON] )
                {

                    // Get the absolute index into secondary buffer
                    ui32 index_level = part_id * parameters.nb_of_secondaries + ( ui32 ) particles.level[ part_id ];

                    // If primary is still alive
                    if ( particles.endsimu[ part_id ] == PARTICLE_ALIVE )
                    {
                        // Store the current particle
                        particles.sec_E[ index_level ]  =  particles.E[ part_id ];
                        particles.sec_px[ index_level ] = particles.px[ part_id ];
                        particles.sec_py[ index_level ] = particles.py[ part_id ];
                        particles.sec_pz[ index_level ] = particles.pz[ part_id ];
                        particles.sec_dx[ index_level ] = particles.dx[ part_id ];
                        particles.sec_dy[ index_level ] = particles.dy[ part_id ];
                        particles.sec_dz[ index_level ] = particles.dz[ part_id ];
                        particles.sec_pname[ index_level ] = particles.pname[ part_id ];
                        // Lose a level in the hierarchy
                        particles.level[ part_id ] += 1;
                    }

                    // Fill the main buffer with the new secondary particle
                    particles.E[ part_id ]  = secondary_part.E;
                    particles.dx[ part_id ] = secondary_part.dir.x;
                    particles.dy[ part_id ] = secondary_part.dir.y;
                    particles.dz[ part_id ] = secondary_part.dir.z;
                    particles.pname[ part_id ] = secondary_part.pname;
                    particles.endsimu[ part_id ] = secondary_part.endsimu;

                    return;

                }
                else
                {

                    // This secondary particle is not used, so drop its energy
                    if ( secondary_part.E != 0.0f )
                    {
                        dose_record_standard( dosi, secondary_part.E, particles.px[ part_id ],
                                              particles.py[ part_id ], particles.pz[ part_id ] );

                    }


#ifdef DEBUG
                    if ( particles.level[ part_id ] == parameters.nb_of_secondaries )
                    {
                        printf( "[ERROR] track_electron_to_out: reach max secondary level\n");
                    }
#endif
                }

                alongStepLength = 0;
                freeLength = 0.;
                totalLength += trueStepLength;

            } // significant_loss == false

        } // bool_loop == true

#ifdef DEBUG
        if ( istep > 1000 )
        {
            printf( "[ERROR] track_electron_to_out: e- reach 1000 steps\n" );
            printf("         ID %i E %e keV - level %i - pos %f %f %f\n", part_id, particles.E[part_id]/keV, particles.level[part_id],
                   particles.px[part_id], particles.py[part_id], particles.pz[part_id] );
            particles.endsimu[part_id] = PARTICLE_DEAD;
            return;
        }

        ++istep;
#endif

        /// DEBUG - FIRST STEP ////////////////////////////////
        //particles.endsimu[ part_id ] = PARTICLE_DEAD;
        //return;
        ///////////////////////////////////////////////////////

#ifdef DEBUG_TRACK_ID

        if ( part_id == DEBUG_TRACK_ID )
        {
           printf("  ID %i - istep %i - Electron - level %i - E %f keV - pos %f %f %f\n",
                  part_id, istep, particles.level[part_id], particles.E[part_id]/keV,
                  particles.px[ part_id ], particles.py[ part_id ], particles.pz[ part_id ] );
        }

#endif


    }
    while ( ( particles.E[ part_id ] > electronEcut ) && ( bool_loop ) );


    //printf("Tracklength %e    step %i\n", totalLength, istep);

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance ( make_f32xyz( particles.px[ part_id ], particles.py[ part_id ], particles.pz[ part_id ] ),
                                           vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax, parameters.geom_tolerance ) )
    {
        particles.endsimu[ part_id ] = PARTICLE_FREEZE;
        return;
    }

    ////////////////////////////////////
    //                            EKINELIMIT
    if ( ( particles.E[part_id] > electronEcut ) /*&&(secondaryParticleCreated == FALSE)*/ ) //>1eV
    {

        ui8 next_discrete_process ;
        ui32 table_index; // index of cross section table
        f32 next_interaction_distance = FLT_MAX;
        f32 dedxeIoni = 0;
        f32 dedxeBrem = 0;
        f32 erange = 0;
        f32 lambda = 0;
        //         bool significant_loss;
        //         f32 edep;
        //         f32 trueGeomLength;
        //         f32 safety;

        // Read position
        f32xyz pos; // mm
        pos.x = particles.px[part_id];
        pos.y = particles.py[part_id];
        pos.z = particles.pz[part_id];

        // Read direction
        f32xyz dir;
        dir.x = particles.dx[part_id];
        dir.y = particles.dy[part_id];
        dir.z = particles.dz[part_id];

        // Get energy
        f32 energy = particles.E[part_id];

        // Defined index phantom
        f32xyz ivoxsize;
        ivoxsize.x = 1.0 / vol.spacing_x;
        ivoxsize.y = 1.0 / vol.spacing_y;
        ivoxsize.z = 1.0 / vol.spacing_z;
        ui32xyzw index_phantom;
        index_phantom.x = ui32 ( ( pos.x + vol.off_x ) * ivoxsize.x );
        index_phantom.y = ui32 ( ( pos.y + vol.off_y ) * ivoxsize.y );
        index_phantom.z = ui32 ( ( pos.z + vol.off_z ) * ivoxsize.z );
        index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                + index_phantom.y*vol.nb_vox_x
                + index_phantom.x; // linear index

#ifdef DEBUG
        assert( index_phantom.w < vol.number_of_voxels );
#endif

        //Get mat index
        ui16 mat_id = vol.values[index_phantom.w];

#ifdef DEBUG
        assert( mat_id < 65536 );
#endif

        //// Get the next distance boundary volume /////////////////////////////////

        // get voxel params
        f32 vox_xmin = index_phantom.x*vol.spacing_x - vol.off_x;
        f32 vox_ymin = index_phantom.y*vol.spacing_y - vol.off_y;
        f32 vox_zmin = index_phantom.z*vol.spacing_z - vol.off_z;
        f32 vox_xmax = vox_xmin + vol.spacing_x;
        f32 vox_ymax = vox_ymin + vol.spacing_y;
        f32 vox_zmax = vox_zmin + vol.spacing_z;

        // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
        // TODO: In theory this have to be applied just at the entry of the particle within the volume
        //       in order to avoid particle entry between voxels. Then, computing improvement can be made
        //       by calling this function only once, just for the particle step=0.    - JB
        pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                                vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters.geom_tolerance );

        // Get distance to edge of voxel
        f32 fragment = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                      vox_ymin, vox_ymax, vox_zmin, vox_zmax );

#ifdef DEBUG
        assert( fragment < FLT_MAX );
#endif

        // fragment += 1.E-2*mm;  ?? - JB
        fragment += parameters.geom_tolerance;

        // Read Cross section table to get dedx, erange, lambda
        e_read_CS_table ( mat_id, energy, electron_CS_table, next_discrete_process, table_index, next_interaction_distance,
                          dedxeIoni,dedxeBrem,erange, lambda, randomnumbereBrem, randomnumbereIoni, parameters );

        f32 cutstep = StepFunction ( erange );

        trueStepLength = GlobalMscScattering ( fragment, cutstep, erange, energy, lambda,   dedxeIoni,
                                               dedxeBrem,  electron_CS_table,  mat_id, particles,  part_id,     // HERE the particle move
                                               par1, par2, materials, dosi, index_phantom, vol, parameters );

        freeLength = alongStepLength + trueStepLength;
        totalLength += trueStepLength;

/*
        /// Need to check, I add energy cut here - JB /////////////////////////////
        if ( particles.E[ part_id ] <= materials.electron_energy_cut[ mat_id ] )
        {
            particles.endsimu[ part_id ] = PARTICLE_DEAD;
            dose_record_standard( dosi, particles.E[ part_id ], particles.px[ part_id ],
                                  particles.py[ part_id ], particles.pz[ part_id ] );

            //printf("  ID %i  Sec last cutE\n", part_id);

            return;
        }
*/
        ///////////////////////////////////////////////////////////////////////////

    }
    else
    {
        // Kill the particle
        particles.endsimu[ part_id ] = PARTICLE_DEAD;

        /// HERE, energy is not droppping ?   - JB   // TO BE CHECKED ////////////
        dose_record_standard( dosi, particles.E[ part_id ], particles.px[ part_id ],
                              particles.py[ part_id ], particles.pz[ part_id ] );
        //////////////////////////////////////////////////////////////////////////


        return;
    }


    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance ( make_f32xyz( particles.px[ part_id ], particles.py[ part_id ], particles.pz[ part_id ] ),
                                           vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax, parameters.geom_tolerance ) )
    {
        particles.endsimu[ part_id ] = PARTICLE_FREEZE;

    }


}


__host__ __device__ void VPDN::track_photon_to_out ( ParticlesData &particles,
                                                     VoxVolumeData vol,
                                                     MaterialsTable materials,
                                                     PhotonCrossSectionTable photon_CS_table,
                                                     GlobalSimulationParametersData parameters,
                                                     DoseData dosi,
                                                     ui32 part_id )
{        
    // Read position
    f32xyz pos;
    pos.x = particles.px[part_id];
    pos.y = particles.py[part_id];
    pos.z = particles.pz[part_id];

    // Read direction
    f32xyz dir;
    dir.x = particles.dx[part_id];
    dir.y = particles.dy[part_id];
    dir.z = particles.dz[part_id];

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / vol.spacing_x;
    ivoxsize.y = 1.0 / vol.spacing_y;
    ivoxsize.z = 1.0 / vol.spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( pos.x + vol.off_x ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( pos.y + vol.off_y ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pos.z + vol.off_z ) * ivoxsize.z );

    index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                      + index_phantom.y*vol.nb_vox_x
                      + index_phantom.x; // linear index

    // Get the material that compose this volume
    ui16 mat_id = vol.values[ index_phantom.w ];

    //// Find next discrete interaction ///////////////////////////////////////

    photon_get_next_interaction ( particles, parameters, photon_CS_table, mat_id, part_id );

    f32 next_interaction_distance = particles.next_interaction_distance[part_id];
    ui8 next_discrete_process = particles.next_discrete_process[part_id];

    //// Get the next distance boundary volume /////////////////////////////////

    f32 vox_xmin = index_phantom.x*vol.spacing_x - vol.off_x;
    f32 vox_ymin = index_phantom.y*vol.spacing_y - vol.off_y;
    f32 vox_zmin = index_phantom.z*vol.spacing_z - vol.off_z;
    f32 vox_xmax = vox_xmin + vol.spacing_x;
    f32 vox_ymax = vox_ymin + vol.spacing_y;
    f32 vox_zmax = vox_zmin + vol.spacing_z;

    // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
    // TODO: In theory this have to be applied just at the entry of the particle within the volume
    //       in order to avoid particle entry between voxels. Then, computing improvement can be made
    //       by calling this function only once, just for the particle step=0.    - JB
    pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                            vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters.geom_tolerance );

    f32 boundary_distance = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                           vox_ymin, vox_ymax, vox_zmin, vox_zmax );

    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters.geom_tolerance; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // get safety position (outside the current voxel)
    pos = transport_get_safety_outside_AABB( pos, vox_xmin, vox_xmax,
                                             vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters.geom_tolerance );

    // store new position
    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance (pos, vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax, parameters.geom_tolerance ) )
    {
        particles.endsimu[part_id] = PARTICLE_FREEZE;
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    if ( next_discrete_process != GEOMETRY_BOUNDARY )
    {
        // Resolve discrete process
        SecParticle electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                                 materials, mat_id, part_id );

        /// Energy cut /////////////

        // If gamma particle not enough energy (Energy cut)
        if ( particles.E[ part_id ] <= materials.photon_energy_cut[ mat_id ] )
        {
            // Kill without mercy
            particles.endsimu[ part_id ] = PARTICLE_DEAD;
        }

        // If electron particle not enough energy (Energy cut)
        if ( electron.E <= materials.electron_energy_cut[ mat_id ] )
        {
            // Kill without mercy
            electron.endsimu = PARTICLE_DEAD;
        }

        /// Drope energy ////////////

        // If gamma particle is dead (PE, Compton or energy cut)
        if ( particles.endsimu[ part_id ] == PARTICLE_DEAD &&  particles.E[ part_id ] != 0.0f )
        {
            dose_record_standard( dosi, particles.E[ part_id ], particles.px[ part_id ],
                                  particles.py[ part_id ], particles.pz[ part_id ] );
        }

        // If electron particle is dead (PE, Compton or energy cut)
        if ( electron.endsimu == PARTICLE_DEAD &&  electron.E != 0.0f )
        {
            dose_record_standard( dosi, electron.E, particles.px[ part_id ],
                                  particles.py[ part_id ], particles.pz[ part_id ] );
        }


        /// Handle secondary

        if ( electron.endsimu == PARTICLE_ALIVE )
        {

            // If secondary enable and enough level space
            if ( particles.level[ part_id ] < parameters.nb_of_secondaries && parameters.secondaries_list[ELECTRON] )
            {
                // Get the absolute index into secondary buffer
                ui32 index_level = part_id * parameters.nb_of_secondaries + ( ui32 ) particles.level[ part_id ];

                // If the current gamma is still alive, store it into the buffer
                if ( particles.endsimu[ part_id ] == PARTICLE_ALIVE )
                {
                    particles.sec_E[ index_level ]  =  particles.E[ part_id ];
                    particles.sec_px[ index_level ] = particles.px[ part_id ];
                    particles.sec_py[ index_level ] = particles.py[ part_id ];
                    particles.sec_pz[ index_level ] = particles.pz[ part_id ];
                    particles.sec_dx[ index_level ] = particles.dx[ part_id ];
                    particles.sec_dy[ index_level ] = particles.dy[ part_id ];
                    particles.sec_dz[ index_level ] = particles.dz[ part_id ];
                    particles.sec_pname[ index_level ] = particles.pname[ part_id ];
                    // Lose a level in the hierarchy
                    particles.level[ part_id ] += 1;
                }

                // Fill the main buffer with the new secondary particle
                particles.E[ part_id ]  = electron.E;
                particles.dx[ part_id ] = electron.dir.x;
                particles.dy[ part_id ] = electron.dir.y;
                particles.dz[ part_id ] = electron.dir.z;
                particles.pname[ part_id ] = electron.pname;
                particles.endsimu[ part_id ] = electron.endsimu;


//                printf("ID %i - Sec level %i (push gtrack) pos %e %e %e dir %e %e %e\n", part_id, particles.level[ part_id ], particles.px[ part_id ],
//                       particles.py[ part_id ],particles.pz[ part_id ],particles.dx[ part_id ],
//                       particles.dy[ part_id ],particles.dz[ part_id ]);


            }
            else
            {
                // This secondary is not used, then drop its energy
                dose_record_standard( dosi, electron.E, particles.px[ part_id ],
                                      particles.py[ part_id ], particles.pz[ part_id ] );

//                printf("ID %i -Sec level %i (Not gtrack) pos %e %e %e dir %e %e %e\n", part_id, particles.level[ part_id ], particles.px[ part_id ],
//                       particles.py[ part_id ],particles.pz[ part_id ],particles.dx[ part_id ],
//                       particles.dy[ part_id ],particles.dz[ part_id ]);

            }
        }


    } // discrete process


}

// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPDN::kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                                  f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance )
{  
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, tolerance, id);
}


// Host Kernel that move particles to the voxelized volume boundary
void VPDN::kernel_host_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 part_id )
{       
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, tolerance, part_id);
}

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPDN::kernel_device_track_to_out ( ParticlesData particles,
                                                   VoxVolumeData vol,
                                                   MaterialsTable materials,
                                                   PhotonCrossSectionTable photon_CS_table,
                                                   ElectronsCrossSectionTable electron_CS_table,
                                                   GlobalSimulationParametersData parameters,
                                                   DoseData dosi )
{   
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    

    // For multivoxels navigation
    f32 randomnumbereIoni= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)
    f32 randomnumbereBrem= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)
    f32 freeLength = 0.0*mm;

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {

        if ( particles.pname[id] == PHOTON )
        {

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("Tracking a photon\n");
        }
#endif

            VPDN::track_photon_to_out ( particles, vol, materials, photon_CS_table, parameters, dosi, id );

        }
        else if ( particles.pname[id] == ELECTRON )
        {

#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("Tracking an electron\n");
        }
#endif
            VPDN::track_electron_to_out ( particles, vol, materials, electron_CS_table, parameters, dosi,
                                          randomnumbereIoni, randomnumbereBrem, freeLength, id );


        }

        // Condition if particle is dead and if it was a secondary
        if ( ( ( particles.endsimu[id]==PARTICLE_DEAD ) || ( particles.endsimu[id]==PARTICLE_FREEZE ) ) && ( particles.level[id]>PRIMARY ) )
        {

            /// Pull back the particle stored in the secondary buffer to the main one

            // DEBUG
            //printf(" PULL e-  ID %i Level %i\n", id, particles.level[id]);


            // Wake up the particle
            particles.endsimu[id] = PARTICLE_ALIVE;
            // Earn a higher level
            particles.level[id]  -= 1;
            // Get the absolute index into secondary buffer
            ui32 index_level = id * parameters.nb_of_secondaries + ( ui32 ) particles.level[id];

            // FreeLength must be reinitialized due to voxels navigation (diff mats)
            freeLength = 0.0*mm;
            randomnumbereIoni= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)
            randomnumbereBrem= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)

            // Get back the stored particle into the primary buffer
            particles.E[ id ]     = particles.sec_E[ index_level ]    ;
            particles.px[ id ]    = particles.sec_px[ index_level ]   ;
            particles.py[ id ]    = particles.sec_py[ index_level ]   ;
            particles.pz[ id ]    = particles.sec_pz[ index_level ]   ;
            particles.dx[ id ]    = particles.sec_dx[ index_level ]   ;
            particles.dy[ id ]    = particles.sec_dy[ index_level ]   ;
            particles.dz[ id ]    = particles.sec_dz[ index_level ]   ;
            particles.pname[ id ] = particles.sec_pname[ index_level ];

//            printf("ID %i - Sec level %i (pull device) pos %e %e %e dir %e %e %e INDEX %i \n", id, particles.level[ id ], particles.px[ id ],
//                   particles.py[ id ],particles.pz[ id ], particles.dx[ id ], particles.dy[ id ], particles.dz[ id ], index_level );

        }

        /// DEBUG

    }

    /// DEBUG
    //if ( step > 1 ) printf("ID %i Step %i\n", id, step);

}

// Host kernel that track particles within the voxelized volume until boundary
void VPDN::kernel_host_track_to_out ( ParticlesData particles,
                                      VoxVolumeData vol,
                                      MaterialsTable materials,
                                      PhotonCrossSectionTable photon_CS_table,
                                      ElectronsCrossSectionTable electron_CS_table,
                                      GlobalSimulationParametersData parameters,
                                      DoseData dosi,
                                      ui32 id )
{
    // For multivoxels navigation
    f32 randomnumbereIoni= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)
    f32 randomnumbereBrem= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)
    f32 freeLength = 0.0*mm;

    ui32 step = 0;

    //printf(" -- Start status %i\n", particles.endsimu[id]);

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {

        if ( particles.pname[id] == PHOTON )
        { 
#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("Tracking a photon\n");
        }
#endif

            VPDN::track_photon_to_out ( particles, vol, materials, photon_CS_table, parameters, dosi, id );

        }
        else if ( particles.pname[id] == ELECTRON )
        {
#ifdef DEBUG_TRACK_ID
        if ( id == DEBUG_TRACK_ID )
        {
            printf("Tracking an electron\n");
        }
#endif
            VPDN::track_electron_to_out ( particles, vol, materials, electron_CS_table, parameters, dosi,
                                          randomnumbereIoni, randomnumbereBrem, freeLength, id );
        }

        // Condition if particle is dead and if it was a secondary
        if ( ( ( particles.endsimu[id]==PARTICLE_DEAD ) || ( particles.endsimu[id]==PARTICLE_FREEZE ) ) && ( particles.level[id]>PRIMARY ) )
        {

            /// Pull back the particle stored in the secondary buffer to the main one            

            // Wake up the particle
            particles.endsimu[id] = PARTICLE_ALIVE;
            // Earn a higher level
            particles.level[id]  -= 1;
            // Get the absolute index into secondary buffer
            ui32 index_level = id * parameters.nb_of_secondaries + ( ui32 ) particles.level[id];

            // FreeLength must be reinitialized due to voxels navigation (diff mats)
            freeLength = 0.0*mm;
            randomnumbereIoni= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)
            randomnumbereBrem= -logf ( prng_uniform( &(particles.prng[id]) ) ); // -log(RN)

            // Get back the stored particle into the primary buffer
            particles.E[ id ]     = particles.sec_E[ index_level ]    ;
            particles.px[ id ]    = particles.sec_px[ index_level ]   ;
            particles.py[ id ]    = particles.sec_py[ index_level ]   ;
            particles.pz[ id ]    = particles.sec_pz[ index_level ]   ;
            particles.dx[ id ]    = particles.sec_dx[ index_level ]   ;
            particles.dy[ id ]    = particles.sec_dy[ index_level ]   ;
            particles.dz[ id ]    = particles.sec_dz[ index_level ]   ;
            particles.pname[ id ] = particles.sec_pname[ index_level ];

//            printf("ID %i - Sec level %i (pull host) pos %e %e %e dir %e %e %e INDEX %i\n", id, particles.level[ id ], particles.px[ id ],
//                   particles.py[ id ],particles.pz[ id ], particles.dx[ id ], particles.dy[ id ], particles.dz[ id ], index_level);


        }

        ++step;


    }


}

////:: Privates

bool VoxPhanDosiNav::m_check_mandatory()
{

    if ( m_phantom.data_h.nb_vox_x == 0 || m_phantom.data_h.nb_vox_y == 0 || m_phantom.data_h.nb_vox_z == 0 ||
            m_phantom.data_h.spacing_x == 0 || m_phantom.data_h.spacing_y == 0 || m_phantom.data_h.spacing_z == 0 ||
            m_phantom.list_of_materials.size() == 0 || m_materials_filename.empty() )
    {
        return false;
    }
    else
    {
        return true;
    }

}

// return memory usage
ui64 VoxPhanDosiNav::m_get_memory_usage()
{
    ui64 mem = 0;

    // First the voxelized phantom
    mem += ( m_phantom.data_h.number_of_voxels * sizeof( ui16 ) );
    // Then material data
    mem += ( ( 2 * m_materials.data_h.nb_elements_total + 23 * m_materials.data_h.nb_materials ) * sizeof( f32 ) );
    // Then cross sections (gamma)
    ui64 n = m_cross_sections.photon_CS.data_h.nb_bins;
    ui64 k = m_cross_sections.photon_CS.data_h.nb_mat;
    mem += ( ( n + 3*n*k + 3*101*n ) * sizeof( f32 ) );
    // Cross section (electron)
    mem += ( n*k*7*sizeof( f32 ) );
    // Finally the dose map
    n = m_dose_calculator.dose.data_h.tot_nb_doxels;
    mem += ( 4*n*sizeof( f64 ) + n*sizeof( ui32 ) );
    mem += ( 20 * sizeof(f32) );

    return mem;
}

////:: Main functions

VoxPhanDosiNav::VoxPhanDosiNav ()
{
    // Default doxel size (if 0 = same size to the phantom)
    m_doxel_size_x = 0;
    m_doxel_size_y = 0;
    m_doxel_size_z = 0;

    m_xmin = 0.0; m_xmax = 0.0;
    m_ymin = 0.0; m_ymax = 0.0;
    m_zmin = 0.0; m_zmax = 0.0;

    m_materials_filename = "";
}

void VoxPhanDosiNav::track_to_in ( Particles particles )
{

    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id=0;
        while ( id<particles.size )
        {
            VPDN::kernel_host_track_to_in ( particles.data_h, m_phantom.data_h.xmin, m_phantom.data_h.xmax,
                                            m_phantom.data_h.ymin, m_phantom.data_h.ymax,
                                            m_phantom.data_h.zmin, m_phantom.data_h.zmax,
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

        VPDN::kernel_device_track_to_in<<<grid, threads>>> ( particles.data_d, m_phantom.data_d.xmin, m_phantom.data_d.xmax,
                                                                               m_phantom.data_d.ymin, m_phantom.data_d.ymax,
                                                                               m_phantom.data_d.zmin, m_phantom.data_d.zmax,
                                                                               m_params.data_d.geom_tolerance );
        cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to in)" );
        cudaThreadSynchronize();
    }

    // DEBUG
    //printf("TrackToIn  ok\n");

}

void VoxPhanDosiNav::track_to_out ( Particles particles )
{
    //
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {

        ui32 id=0;
        while ( id<particles.size )
        {

            // DEBUG
            //printf("TrackToOut id %i\n", id);
            VPDN::kernel_host_track_to_out ( particles.data_h, m_phantom.data_h,
                                             m_materials.data_h, m_cross_sections.photon_CS.data_h, m_cross_sections.electron_CS.data_h,
                                             m_params.data_h, m_dose_calculator.dose.data_h, id );

            ++id;
        }
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {       
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;//
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;
        cudaThreadSynchronize();
        VPDN::kernel_device_track_to_out<<<grid, threads>>> ( particles.data_d, m_phantom.data_d, m_materials.data_d,
                                                              m_cross_sections.photon_CS.data_d,
                                                              m_cross_sections.electron_CS.data_d,
                                                              m_params.data_d, m_dose_calculator.dose.data_d );
        cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to out)" );
        
        cudaThreadSynchronize();
    }
    
    
}

void VoxPhanDosiNav::load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
{
    m_phantom.load_from_mhd ( filename, range_mat_name );
}

void VoxPhanDosiNav::write ( std::string filename )
{
//     m_dose_calculator.m_copy_dose_gpu2cpu();

    m_dose_calculator.write ( filename );
}


void VoxPhanDosiNav::initialize ( GlobalSimulationParameters params )
{
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanDosi: missing parameters." );
        exit_simulation();
    }

    // Params
    m_params = params;

    // Phantom
    m_phantom.set_name ( "VoxPhanDosiNav" );
    m_phantom.initialize ( params );

    // Materials table
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize ( m_phantom.list_of_materials, params );

    // Cross Sections
    m_cross_sections.initialize ( m_materials, params );

    // Init dose map
    m_dose_calculator.set_voxelized_phantom( m_phantom );
    m_dose_calculator.set_materials( m_materials );
    m_dose_calculator.set_doxel_size( m_doxel_size_x, m_doxel_size_y, m_doxel_size_z );
    m_dose_calculator.set_voi( m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax );
    m_dose_calculator.initialize ( m_params ); // CPU&GPU
    
    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanDosiNav", mem);
    }

}

void VoxPhanDosiNav::calculate_dose_to_water(){

    m_dose_calculator.calculate_dose_to_water();

}

void VoxPhanDosiNav::calculate_dose_to_phantom(){

    m_dose_calculator.calculate_dose_to_phantom();

}

void VoxPhanDosiNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}

void VoxPhanDosiNav::set_doxel_size( f32 sizex, f32 sizey, f32 sizez )
{
    m_doxel_size_x = sizex;
    m_doxel_size_y = sizey;
    m_doxel_size_z = sizez;
}


#undef DEBUG

#endif
