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

__host__ __device__ void VPDN::track_electron_to_out( ParticlesData *particles, ParticlesData *buffer,
                                                      const VoxVolumeData<ui16> *vol,
                                                      const MaterialsData *materials,
                                                      const ElectronsCrossSectionData *electron_CS_table,
                                                      const GlobalSimulationParametersData *parameters,
                                                      DoseData *dosi,
                                                      f32 &randomnumbereIoni,
                                                      f32 &randomnumbereBrem,
                                                      f32 &freeLength,
                                                      ui32 part_id )
{

    // Parameters values need to be stored for every e-step
    f32 alongStepLength = 0.;               // Distance from the last physics interaction.
    bool lastStepisaPhysicEffect = TRUE;    // To store last random number
    //bool bool_loop = true;                // If it is not the last step in the same voxel

    alongStepLength = freeLength;
    if ( freeLength > 0.0 ) lastStepisaPhysicEffect = FALSE; // Changement de voxel sans effet physique

    // Parameters
    f32 trueStepLength = FLT_MAX;
    //f32 totalLength = 0.;
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
        if ( !test_point_AABB_with_tolerance ( make_f32xyz( particles->px[ part_id ], particles->py[ part_id ], particles->pz[ part_id ] ),
                                               vol->xmin, vol->xmax, vol->ymin, vol->ymax, vol->zmin, vol->zmax, parameters->geom_tolerance ) )
        {
            particles->status[ part_id ] = PARTICLE_FREEZE;
            return;
        }

        // Get Random number stored until a physic interaction
        if ( lastStepisaPhysicEffect == TRUE )
        {
            randomnumbereBrem = -logf ( prng_uniform( particles, part_id ) );
            randomnumbereIoni = -logf ( prng_uniform( particles, part_id ) );
            alongStepLength = 0.f;
            lastStepisaPhysicEffect = FALSE;
        }

        // Read position
        pos.x = particles->px[part_id];
        pos.y = particles->py[part_id];
        pos.z = particles->pz[part_id];

        // Read direction
        dir.x = particles->dx[part_id];
        dir.y = particles->dy[part_id];
        dir.z = particles->dz[part_id];

        // Read energy
        energy = particles->E[part_id];

        // Defined index phantom
        f32xyz ivoxsize;
        ivoxsize.x = 1.0 / vol->spacing_x;
        ivoxsize.y = 1.0 / vol->spacing_y;
        ivoxsize.z = 1.0 / vol->spacing_z;
        ui32xyzw index_phantom;
        index_phantom.x = ui32 ( ( pos.x + vol->off_x ) * ivoxsize.x );
        index_phantom.y = ui32 ( ( pos.y + vol->off_y ) * ivoxsize.y );
        index_phantom.z = ui32 ( ( pos.z + vol->off_z ) * ivoxsize.z );

        index_phantom.w = index_phantom.z*vol->nb_vox_x*vol->nb_vox_y
                + index_phantom.y*vol->nb_vox_x
                + index_phantom.x; // linear index

#ifdef DEBUG
        assert( index_phantom.w < vol->number_of_voxels );
#endif

        // Get the material that compose this volume
        ui16 mat_id = vol->values[ index_phantom.w ];
        electronEcut = materials->electron_energy_cut[ mat_id ];

#ifdef DEBUG
        assert( mat_id < 65536 );
#endif

        // Read the different CS, dE/dx tables
        e_read_CS_table( mat_id, energy, electron_CS_table, next_discrete_process, table_index,
                         next_interaction_distance, dedxeIoni, dedxeBrem, erange, lambda, randomnumbereBrem, randomnumbereIoni, parameters );

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
        f32 vox_xmin = index_phantom.x*vol->spacing_x - vol->off_x;
        f32 vox_ymin = index_phantom.y*vol->spacing_y - vol->off_y;
        f32 vox_zmin = index_phantom.z*vol->spacing_z - vol->off_z;
        f32 vox_xmax = vox_xmin + vol->spacing_x;
        f32 vox_ymax = vox_ymin + vol->spacing_y;
        f32 vox_zmax = vox_zmin + vol->spacing_z;

        // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
        // TODO: In theory this have to be applied just at the entry of the particle within the volume
        //       in order to avoid particle entry between voxels. Then, computing improvement can be made
        //       by calling this function only once, just for the particle step=0.    - JB
        pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                                vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters->geom_tolerance );

        // compute the next distance boundary
        f32 boundary_distance = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                               vox_ymin, vox_ymax, vox_zmin, vox_zmax );

#ifdef DEBUG
        assert( boundary_distance != FLT_MAX );
#endif

        if ( trueStepLength > boundary_distance )
        {
            if ( parameters->physics_list[ELECTRON_MSC] == ENABLED )
            {
                trueGeomLength = gTransformToGeom( trueStepLength, erange, lambda, energy,
                                                    par1, par2, electron_CS_table, mat_id );

                if ( trueGeomLength > boundary_distance )
                {
                    trueStepLength = GlobalMscScattering( boundary_distance + parameters->geom_tolerance, cutstep, erange, energy, lambda,   dedxeIoni,
                                                          dedxeBrem,  electron_CS_table,  mat_id, particles,  part_id,     // HERE the particle move
                                                          par1, par2, materials, dosi, vol, parameters );

                    freeLength = alongStepLength + trueStepLength;

                    // Stop simulation if out of the phantom
                    if ( !test_point_AABB_with_tolerance( make_f32xyz( particles->px[ part_id ], particles->py[ part_id ], particles->pz[ part_id ] ),
                                                          vol->xmin, vol->xmax, vol->ymin, vol->ymax, vol->zmin, vol->zmax, parameters->geom_tolerance ) )
                    {
                        particles->status[ part_id ] = PARTICLE_FREEZE;

                    }
                    return;                                                                                                    
                }
            }
            else
            {
                trueStepLength = GlobalMscScattering( boundary_distance + parameters->geom_tolerance, cutstep, erange, energy, lambda,   dedxeIoni,
                                                      dedxeBrem,  electron_CS_table,  mat_id, particles,  part_id,     // HERE the particle move
                                                      par1, par2, materials, dosi, vol, parameters );

                freeLength = alongStepLength + trueStepLength;

                // Stop simulation if out of the phantom
                if ( !test_point_AABB_with_tolerance( make_f32xyz( particles->px[ part_id ], particles->py[ part_id ], particles->pz[ part_id ] ),
                                                      vol->xmin, vol->xmax, vol->ymin, vol->ymax, vol->zmin, vol->zmax, parameters->geom_tolerance ) )
                {
                    particles->status[ part_id ] = PARTICLE_FREEZE;

#ifdef DEBUG_TRACK_ID
                        if ( part_id == DEBUG_TRACK_ID )
                        {
                            printf("  ID %i Tracking an electron OUT OF PHANTOM\n", part_id );
                        }
#endif
                }

                return;
            }

        }


        if ( significant_loss == true )
        {

            // Energy loss (call eFluctuation)
            edep = eLoss( trueStepLength, particles->E[ part_id ], dedxeIoni, dedxeBrem, erange,
                          electron_CS_table, mat_id, materials, particles, part_id );

            GlobalMscScattering( trueStepLength, cutstep, erange, energy, lambda, dedxeIoni,
                                 dedxeBrem,  electron_CS_table,  mat_id, particles, part_id, par1, par2,    // HERE particle move
                                 materials, dosi, vol, parameters );

            dose_record_standard( dosi, edep, particles->px[ part_id ], particles->py[ part_id ], particles->pz[ part_id ] );

            alongStepLength += trueStepLength;
            //totalLength += trueStepLength;
            lastStepisaPhysicEffect = FALSE;
        }
        else
        {

            //// InvokeAlongStepDoItProcs ////////////////////////////////////////////////////////////////////////

            // Energy loss (call eFluctuation)
            edep = eLoss( trueStepLength, particles->E[ part_id ], dedxeIoni, dedxeBrem, erange,
                          electron_CS_table, mat_id, materials, particles, part_id );

            GlobalMscScattering( trueStepLength, lengthtoVertex, erange, energy, lambda,   dedxeIoni,
                                 dedxeBrem,   electron_CS_table,  mat_id, particles,  part_id, par1, par2,     // HERE particle move
                                 materials, dosi, vol, parameters );

            dose_record_standard( dosi, edep, particles->px[part_id], particles->py[part_id], particles->pz[part_id] );

            //// InvokePostStepDoItProcs ////////////////////////////////////////////////////////////////////////

            SecParticle secondary_part;
            secondary_part.E = 0.;
            secondary_part.endsimu = PARTICLE_DEAD;

            if ( next_discrete_process == ELECTRON_IONISATION )
            {

                secondary_part = eSampleSecondarieElectron( electronEcut, particles,  part_id );
                lastStepisaPhysicEffect = TRUE;

            }
            else if ( next_discrete_process == ELECTRON_BREMSSTRAHLUNG )
            {
                /// TODO return a photon - JB
                eSampleSecondarieGamma( parameters->cs_table_min_E, parameters->cs_table_max_E, particles,
                                        part_id, materials, mat_id );
                lastStepisaPhysicEffect = TRUE;
            }

            /// If there is a secondary particle, push the primary into buffer and track this new particle

            /// Handle secondary //////////////////////

            if ( secondary_part.endsimu == PARTICLE_ALIVE &&
                 particles->level[ part_id ] < parameters->nb_of_secondaries && parameters->secondaries_list[ELECTRON] )
            {
                // Get the absolute index into secondary buffer
                ui32 index_level = part_id * parameters->nb_of_secondaries + ( ui32 ) particles->level[ part_id ];

                // If primary is still alive
                if ( particles->status[ part_id ] == PARTICLE_ALIVE )
                {
                    // Store the current particle
                    buffer->E[ index_level ]  =  particles->E[ part_id ];
                    buffer->px[ index_level ] = particles->px[ part_id ];
                    buffer->py[ index_level ] = particles->py[ part_id ];
                    buffer->pz[ index_level ] = particles->pz[ part_id ];
                    buffer->dx[ index_level ] = particles->dx[ part_id ];
                    buffer->dy[ index_level ] = particles->dy[ part_id ];
                    buffer->dz[ index_level ] = particles->dz[ part_id ];
                    buffer->pname[ index_level ] = particles->pname[ part_id ];
                    buffer->tof[ index_level ] = particles->tof[ part_id ];
                    // Lose a level in the hierarchy
                    particles->level[ part_id ] += 1;
                }

                // Fill the main buffer with the new secondary particle
                particles->E[ part_id ]  = secondary_part.E;
                particles->dx[ part_id ] = secondary_part.dir.x;
                particles->dy[ part_id ] = secondary_part.dir.y;
                particles->dz[ part_id ] = secondary_part.dir.z;
                particles->pname[ part_id ] = secondary_part.pname;
                particles->status[ part_id ] = secondary_part.endsimu;
                particles->tof[ part_id ] = 0.0;

                alongStepLength = 0;
                freeLength = 0.;

                return;

            }
            else
            {
                // This secondary particle is not used, so drop its energy
                if ( secondary_part.E != 0.0f )
                {
                    dose_record_standard( dosi, secondary_part.E, particles->px[ part_id ],
                                          particles->py[ part_id ], particles->pz[ part_id ] );
                }

#ifdef DEBUG
                if ( particles->level[ part_id ] == parameters->nb_of_secondaries )
                {
                    printf( "[ERROR] track_electron_to_out: reach max secondary level\n");
                }
#endif
            }

            //totalLength += trueStepLength;

        } // significant_loss == false

#ifdef DEBUG
        if ( istep > 1000 )
        {
            printf( "[ERROR] track_electron_to_out: e- reach 1000 steps\n" );
            printf("         ID %i E %e keV - level %i - pos %f %f %f\n", part_id, particles->E[part_id]/keV, particles->level[part_id],
                        particles->px[part_id], particles->py[part_id], particles->pz[part_id] );
                        particles->status[part_id] = PARTICLE_DEAD;
            return;
        }

        ++istep;
#endif

    }
    while ( particles->E[ part_id ] > electronEcut );

    // Kill the particle
    particles->status[ part_id ] = PARTICLE_DEAD;

    /// HERE, energy is not droppping ?   - JB   // TO BE CHECKED ////////////
    dose_record_standard( dosi, particles->E[ part_id ], particles->px[ part_id ],
                          particles->py[ part_id ], particles->pz[ part_id ] );
    //////////////////////////////////////////////////////////////////////////

}


__host__ __device__ void VPDN::track_photon_to_out( ParticlesData *particles,
                                                    ParticlesData *buffer,
                                                    const VoxVolumeData<ui16> *vol,
                                                    const MaterialsData *materials,
                                                    const PhotonCrossSectionData *photon_CS_table,
                                                    const GlobalSimulationParametersData *parameters,
                                                    DoseData *dosi,
                                                    ui32 part_id )
{        
    // Read position
    f32xyz pos;
    pos.x = particles->px[part_id];
    pos.y = particles->py[part_id];
    pos.z = particles->pz[part_id];

    // Read direction
    f32xyz dir;
    dir.x = particles->dx[part_id];
    dir.y = particles->dy[part_id];
    dir.z = particles->dz[part_id];

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / vol->spacing_x;
    ivoxsize.y = 1.0 / vol->spacing_y;
    ivoxsize.z = 1.0 / vol->spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( pos.x + vol->off_x ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( pos.y + vol->off_y ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pos.z + vol->off_z ) * ivoxsize.z );

    index_phantom.w = index_phantom.z*vol->nb_vox_x*vol->nb_vox_y
                      + index_phantom.y*vol->nb_vox_x
                      + index_phantom.x; // linear index

    // Get the material that compose this volume
    ui16 mat_id = vol->values[ index_phantom.w ];

    //// Find next discrete interaction ///////////////////////////////////////

    photon_get_next_interaction ( particles, parameters, photon_CS_table, mat_id, part_id );

    f32 next_interaction_distance = particles->next_interaction_distance[part_id];
    ui8 next_discrete_process = particles->next_discrete_process[part_id];

    //// Get the next distance boundary volume /////////////////////////////////

    f32 vox_xmin = index_phantom.x*vol->spacing_x - vol->off_x;
    f32 vox_ymin = index_phantom.y*vol->spacing_y - vol->off_y;
    f32 vox_zmin = index_phantom.z*vol->spacing_z - vol->off_z;
    f32 vox_xmax = vox_xmin + vol->spacing_x;
    f32 vox_ymax = vox_ymin + vol->spacing_y;
    f32 vox_zmax = vox_zmin + vol->spacing_z;

    // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
    // TODO: In theory this have to be applied just at the entry of the particle within the volume
    //       in order to avoid particle entry between voxels. Then, computing improvement can be made
    //       by calling this function only once, just for the particle step=0.    - JB
    pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                            vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters->geom_tolerance );

    f32 boundary_distance = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                           vox_ymin, vox_ymax, vox_zmin, vox_zmax );

    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters->geom_tolerance; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add( pos, fxyz_scale( dir, next_interaction_distance ) );

    // get safety position (outside the current voxel)
    pos = transport_get_safety_outside_AABB( pos, vox_xmin, vox_xmax,
                                             vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters->geom_tolerance );

    // store new position
    particles->px[part_id] = pos.x;
    particles->py[part_id] = pos.y;
    particles->pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance (pos, vol->xmin, vol->xmax, vol->ymin, vol->ymax, vol->zmin, vol->zmax, parameters->geom_tolerance ) )
    {
        particles->status[part_id] = PARTICLE_FREEZE;
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
        if ( particles->E[ part_id ] <= materials->photon_energy_cut[ mat_id ] )
        {
            // Kill without mercy
            particles->status[ part_id ] = PARTICLE_DEAD;
        }

        // If electron particle not enough energy (Energy cut)
        if ( electron.E <= materials->electron_energy_cut[ mat_id ] )
        {
            // Kill without mercy
            electron.endsimu = PARTICLE_DEAD;
        }

        /// Drope energy ////////////

        // If gamma particle is dead (PE, Compton or energy cut)
        if ( particles->status[ part_id ] == PARTICLE_DEAD &&  particles->E[ part_id ] != 0.0f )
        {
            dose_record_standard( dosi, particles->E[ part_id ], particles->px[ part_id ],
                                  particles->py[ part_id ], particles->pz[ part_id ] );
        }

        // If electron particle is dead (PE, Compton or energy cut)
        if ( electron.endsimu == PARTICLE_DEAD &&  electron.E != 0.0f )
        {
            dose_record_standard( dosi, electron.E, particles->px[ part_id ],
                                  particles->py[ part_id ], particles->pz[ part_id ] );
        }


        /// Handle secondary

        if ( electron.endsimu == PARTICLE_ALIVE )
        {
            // If secondary enable and enough level space
            if ( particles->level[ part_id ] < parameters->nb_of_secondaries && parameters->secondaries_list[ELECTRON] )
            {
                // Get the absolute index into secondary buffer
                ui32 index_level = part_id * parameters->nb_of_secondaries + ( ui32 ) particles->level[ part_id ];

                // If the current gamma is still alive, store it into the buffer
                if ( particles->status[ part_id ] == PARTICLE_ALIVE )
                {
                    buffer->E[ index_level ]  =  particles->E[ part_id ];
                    buffer->px[ index_level ] = particles->px[ part_id ];
                    buffer->py[ index_level ] = particles->py[ part_id ];
                    buffer->pz[ index_level ] = particles->pz[ part_id ];
                    buffer->dx[ index_level ] = particles->dx[ part_id ];
                    buffer->dy[ index_level ] = particles->dy[ part_id ];
                    buffer->dz[ index_level ] = particles->dz[ part_id ];
                    buffer->pname[ index_level ] = particles->pname[ part_id ];
                    buffer->tof[ index_level ] = particles->tof[ part_id ];
                    // Lose a level in the hierarchy
                    particles->level[ part_id ] += 1;
                }

                // Fill the main buffer with the new secondary particle
                particles->E[ part_id ]  = electron.E;
                particles->dx[ part_id ] = electron.dir.x;
                particles->dy[ part_id ] = electron.dir.y;
                particles->dz[ part_id ] = electron.dir.z;
                particles->pname[ part_id ] = electron.pname;
                particles->tof[ part_id ] = 0.0;
                particles->status[ part_id ] = electron.endsimu;

            }
            else
            {
                // This secondary is not used, then drop its energy
                dose_record_standard( dosi, electron.E, particles->px[ part_id ],
                                      particles->py[ part_id ], particles->pz[ part_id ] );
            }
        }


    } // discrete process


}

// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPDN::kernel_device_track_to_in ( ParticlesData *particles, f32 xmin, f32 xmax,
                                                  f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance )
{  
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles->size ) return;

    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, tolerance, id );
}

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPDN::kernel_device_track_to_out( ParticlesData *particles, ParticlesData *buffer,
                                                  const VoxVolumeData<ui16> *vol,
                                                  const MaterialsData *materials,
                                                  const PhotonCrossSectionData *photon_CS_table,
                                                  const ElectronsCrossSectionData *electron_CS_table,
                                                  const GlobalSimulationParametersData *parameters,
                                                  DoseData *dosi )
{   

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles->size ) return;

    // For multivoxels navigation
    f32 randomnumbereIoni= -logf ( prng_uniform( particles, id ) ); // -log(RN)
    f32 randomnumbereBrem= -logf ( prng_uniform( particles, id ) ); // -log(RN)
    f32 freeLength = 0.0*mm;

#ifdef DEBUG
    ui32 iter = 0;
#endif

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles->status[id] != PARTICLE_DEAD && particles->status[id] != PARTICLE_FREEZE )
    {

        if ( particles->pname[id] == PHOTON )
        {
            VPDN::track_photon_to_out( particles, buffer, vol, materials, photon_CS_table, parameters, dosi, id );
        }

        else if ( particles->pname[id] == ELECTRON )
        {
            VPDN::track_electron_to_out( particles, buffer, vol, materials, electron_CS_table, parameters, dosi,
                                         randomnumbereIoni, randomnumbereBrem, freeLength, id );
        }

        // Condition if particle is dead and if it was a secondary
        if ( ( ( particles->status[id] == PARTICLE_DEAD ) || ( particles->status[id] == PARTICLE_FREEZE ) )
             && ( particles->level[id] > PRIMARY ) )
        {
            /// Pull back the particle stored in the secondary buffer to the main one

            // Wake up the particle
            particles->status[id] = PARTICLE_ALIVE;

            // Decrease the level
            particles->level[id]  -= 1;

            // Get the absolute index into secondary buffer
            ui32 index_level = id * parameters->nb_of_secondaries + ( ui32 ) particles->level[id];

            // FreeLength must be reinitialized due to voxels navigation (diff mats)
            freeLength = 0.0*mm;
            randomnumbereIoni= -logf ( prng_uniform( particles, id ) ); // -log(RN)
            randomnumbereBrem= -logf ( prng_uniform( particles, id ) ); // -log(RN)

            // Get back the stored particle into the primary buffer
            particles->E[ id ]     = buffer->E[ index_level ]    ;
            particles->px[ id ]    = buffer->px[ index_level ]   ;
            particles->py[ id ]    = buffer->py[ index_level ]   ;
            particles->pz[ id ]    = buffer->pz[ index_level ]   ;
            particles->dx[ id ]    = buffer->dx[ index_level ]   ;
            particles->dy[ id ]    = buffer->dy[ index_level ]   ;
            particles->dz[ id ]    = buffer->dz[ index_level ]   ;
            particles->tof[ id ]   = buffer->tof[ index_level ]  ;
            particles->pname[ id ] = buffer->pname[ index_level ];
        }


#ifdef DEBUG
        ++iter;

        if ( iter > 10000 )
        {
            printf("DEBUG MODE ID %i: inf loop in particle (photon and electron) stepping\n", id);
            return;
        }
#endif

    }

}


////:: Privates

bool VoxPhanDosiNav::m_check_mandatory()
{

    if ( m_phantom.h_volume->nb_vox_x == 0 || m_phantom.h_volume->nb_vox_y == 0 || m_phantom.h_volume->nb_vox_z == 0 ||
         m_phantom.h_volume->spacing_x == 0 || m_phantom.h_volume->spacing_y == 0 || m_phantom.h_volume->spacing_z == 0 ||
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
    mem += ( m_phantom.h_volume->number_of_voxels * sizeof( ui16 ) );
    // Then material data
    mem += ( ( 3 * m_materials.h_materials->nb_elements_total + 23 * m_materials.h_materials->nb_materials ) * sizeof( f32 ) );
    // Then cross sections (gamma)
    ui64 n = m_cross_sections.h_photon_CS->nb_bins;
    ui64 k = m_cross_sections.h_photon_CS->nb_mat;
    mem += ( ( n + 3*n*k + 3*101*n ) * sizeof( f32 ) );
    // Cross section (electron)
    mem += ( n*k*7*sizeof( f32 ) );
    // Particles buffer
    n = mh_params->size_of_particles_batch * mh_params->nb_of_secondaries;
    mem += ( n * 8 * sizeof(f32) + n * sizeof(ui8) );
    // Finally the dose map
    n = m_dose_calculator.h_dose->tot_nb_dosels;
    mem += ( 2*n*sizeof( f64 ) + n*sizeof( ui32 ) );
    mem += ( 20 * sizeof(f32) );

    return mem;
}

////:: Main functions

VoxPhanDosiNav::VoxPhanDosiNav ()
{
    // Default dosel size (if 0 = same size to the phantom)
    m_dosel_size_x = 0;
    m_dosel_size_y = 0;
    m_dosel_size_z = 0;

    m_xmin = 0.0; m_xmax = 0.0;
    m_ymin = 0.0; m_ymax = 0.0;
    m_zmin = 0.0; m_zmax = 0.0;

    m_materials_filename = "";

    m_dose_min_density = 0.0;

    mh_params = nullptr;
    md_params = nullptr;

    set_name( "VoxPhanDosiNav" );
}

void VoxPhanDosiNav::track_to_in( ParticlesData *d_particles )
{    
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 ) / mh_params->gpu_block_size;

    VPDN::kernel_device_track_to_in<<<grid, threads>>>( d_particles, m_phantom.d_volume->xmin, m_phantom.d_volume->xmax,
                                                        m_phantom.d_volume->ymin, m_phantom.d_volume->ymax,
                                                        m_phantom.d_volume->zmin, m_phantom.d_volume->zmax,
                                                        mh_params->geom_tolerance );
    cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to in)" );
    cudaDeviceSynchronize();
}

void VoxPhanDosiNav::track_to_out( ParticlesData *d_particles )
{    
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 ) / mh_params->gpu_block_size;

    VPDN::kernel_device_track_to_out<<<grid, threads>>>( d_particles, m_particles_buffer.d_particles,
                                                         m_phantom.d_volume, m_materials.d_materials,
                                                         m_cross_sections.d_photon_CS,
                                                         m_cross_sections.d_electron_CS,
                                                         md_params, m_dose_calculator.d_dose );
    cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to out)" );

    cudaDeviceSynchronize();
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

// Export density values of the phantom
void VoxPhanDosiNav::export_density_map( std::string filename )
{
    ui32 N = m_phantom.h_volume->number_of_voxels;
    f32 *density = new f32[ N ];
    ui32 i = 0;
    while (i < N)
    {
        density[ i ] = m_materials.h_materials->density[ m_phantom.h_volume->values[ i ] ];
        ++i;
    }

    f32xyz offset = make_f32xyz( m_phantom.h_volume->off_x, m_phantom.h_volume->off_y, m_phantom.h_volume->off_z );
    f32xyz voxsize = make_f32xyz( m_phantom.h_volume->spacing_x, m_phantom.h_volume->spacing_y, m_phantom.h_volume->spacing_z );
    ui32xyz nbvox = make_ui32xyz( m_phantom.h_volume->nb_vox_x, m_phantom.h_volume->nb_vox_y, m_phantom.h_volume->nb_vox_z );

    ImageIO *im_io = new ImageIO;
    im_io->write_3D( filename, density, nbvox, offset, voxsize );
    delete im_io;
}

// Export materials index of the phantom
void VoxPhanDosiNav::export_materials_map( std::string filename )
{
    f32xyz offset = make_f32xyz( m_phantom.h_volume->off_x, m_phantom.h_volume->off_y, m_phantom.h_volume->off_z );
    f32xyz voxsize = make_f32xyz( m_phantom.h_volume->spacing_x, m_phantom.h_volume->spacing_y, m_phantom.h_volume->spacing_z );
    ui32xyz nbvox = make_ui32xyz( m_phantom.h_volume->nb_vox_x, m_phantom.h_volume->nb_vox_y, m_phantom.h_volume->nb_vox_z );

    ImageIO *im_io = new ImageIO;
    im_io->write_3D( filename, m_phantom.h_volume->values, nbvox, offset, voxsize );
    delete im_io;

}

void VoxPhanDosiNav::initialize (GlobalSimulationParametersData *h_params , GlobalSimulationParametersData *d_params)
{
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanDosi: missing parameters." );
        exit_simulation();
    }

    // Params
    mh_params = h_params;
    md_params = d_params;

    // Phantom
    m_phantom.set_name( "VoxPhanDosiNav" );
    m_phantom.initialize();

    // Materials table
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize( m_phantom.list_of_materials, h_params );

    // Cross Sections
    m_cross_sections.initialize( m_materials.h_materials, h_params );

    // Init dose map
    m_dose_calculator.set_voxelized_phantom( m_phantom );
    m_dose_calculator.set_materials( m_materials );
    m_dose_calculator.set_dosel_size( m_dosel_size_x, m_dosel_size_y, m_dosel_size_z );
    m_dose_calculator.set_min_density( m_dose_min_density );
    m_dose_calculator.set_voi( m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax );
    m_dose_calculator.initialize( h_params );

    // Init buffer of particles to track secondaries
    m_particles_buffer.initialize_secondaries( h_params );

    // Some verbose if required
    if ( h_params->display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanDosiNav", mem);
    }

}

void VoxPhanDosiNav::calculate_dose_to_water(){

    m_dose_calculator.calculate_dose_to_water();

}

void VoxPhanDosiNav::calculate_dose_to_medium(){

    m_dose_calculator.calculate_dose_to_medium();

}

void VoxPhanDosiNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}

void VoxPhanDosiNav::set_dosel_size( f32 sizex, f32 sizey, f32 sizez )
{
    m_dosel_size_x = sizex;
    m_dosel_size_y = sizey;
    m_dosel_size_z = sizez;
}

void VoxPhanDosiNav::set_dose_min_density( f32 min )
{
    m_dose_min_density = min;
}

void VoxPhanDosiNav::set_volume_of_interest( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{
    m_xmin = xmin; m_xmax = xmax;
    m_ymin = ymin; m_ymax = ymax;
    m_zmin = zmin; m_zmax = zmax;
}

AabbData VoxPhanDosiNav::get_bounding_box()
{
    AabbData box;

    box.xmin = m_phantom.h_volume->xmin;
    box.xmax = m_phantom.h_volume->xmax;
    box.ymin = m_phantom.h_volume->ymin;
    box.ymax = m_phantom.h_volume->ymax;
    box.zmin = m_phantom.h_volume->zmin;
    box.zmax = m_phantom.h_volume->zmax;

    return box;
}

#endif
