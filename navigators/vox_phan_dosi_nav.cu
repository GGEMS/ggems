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
    bool secondaryParticleCreated = FALSE;  // If a secondary particle is created

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


    do
    {

        // Get Random number stored until a physic interaction
        if ( lastStepisaPhysicEffect == TRUE )
        {
            randomnumbereBrem = -logf ( JKISS32 ( particles, part_id ) );
            randomnumbereIoni = -logf ( JKISS32 ( particles, part_id ) );
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
        index_phantom.x = ui32 ( ( pos.x-vol.off_x ) * ivoxsize.x );
        index_phantom.y = ui32 ( ( pos.y-vol.off_y ) * ivoxsize.y );
        index_phantom.z = ui32 ( ( pos.z-vol.off_z ) * ivoxsize.z );

        index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                          + index_phantom.y*vol.nb_vox_x
                          + index_phantom.x; // linear index

        // Get the material that compose this volume
        ui16 mat_id = vol.values[ index_phantom.w ];
        
        // Read the different CS, dE/dx tables
        e_read_CS_table ( mat_id, energy, electron_CS_table, next_discrete_process, table_index,
                          next_interaction_distance, dedxeIoni,dedxeBrem, erange, lambda, randomnumbereBrem, randomnumbereIoni, parameters );

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
        f32 vox_xmin = index_phantom.x*vol.spacing_x + vol.off_x;
        f32 vox_ymin = index_phantom.y*vol.spacing_y + vol.off_y;
        f32 vox_zmin = index_phantom.z*vol.spacing_z + vol.off_z;
        f32 vox_xmax = vox_xmin + vol.spacing_x;
        f32 vox_ymax = vox_ymin + vol.spacing_y;
        f32 vox_zmax = vox_zmin + vol.spacing_z;

        // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
        // TODO: In theory this have to be applied just at the entry of the particle within the volume
        //       in order to avoid particle entry between voxels. Then, computing improvement can be made
        //       by calling this function only once, just for the particle step=0.    - JB
        pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                                vox_ymin, vox_ymax, vox_zmin, vox_zmax, EPSILON6 );

        // compute the next distance boundary
        f32 boundary_distance = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                               vox_ymin, vox_ymax, vox_zmin, vox_zmax );

        if ( boundary_distance < trueStepLength )
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

            // Energy loss (call eFluctuation)
            edep = eLoss ( trueStepLength, particles.E[part_id], dedxeIoni, dedxeBrem, erange,
                           electron_CS_table, mat_id, materials, particles, parameters, part_id );

            // Update energy
            particles.E[ part_id ] -= edep;
            
            if ( significant_loss == true )
            {
                GlobalMscScattering ( trueStepLength, cutstep, erange, energy, lambda, dedxeIoni,
                                      dedxeBrem,  electron_CS_table,  mat_id, particles, part_id, par1, par2,
                                      materials, dosi, index_phantom,vol,parameters );
                
                dose_record_standard ( dosi, edep, particles.px[part_id], particles.py[part_id], particles.pz[part_id] );
                alongStepLength+=trueStepLength;
                totalLength+=trueStepLength;
                lastStepisaPhysicEffect=FALSE;

            }
            else
            {

                SecParticle secondary_part;
                secondary_part.E =0.;

                GlobalMscScattering ( trueStepLength, lengthtoVertex, erange, energy, lambda,   dedxeIoni,  dedxeBrem,   electron_CS_table,  mat_id, particles,  part_id, par1, par2, materials, dosi, index_phantom,vol,parameters );

                dose_record_standard ( dosi, edep, particles.px[part_id], particles.py[part_id], particles.pz[part_id] );
                //

                if ( next_discrete_process == ELECTRON_IONISATION )
                {

                    secondary_part = eSampleSecondarieElectron ( parameters.electron_cut, particles,  part_id, dosi,parameters );
                    lastStepisaPhysicEffect=TRUE;

                }
                else if ( next_discrete_process == ELECTRON_BREMSSTRAHLUNG )
                {

                    eSampleSecondarieGamma ( parameters.photon_cut, particles, part_id, materials, mat_id,parameters );
                    lastStepisaPhysicEffect=TRUE;

                }


                // Add primary to buffer an track secondary
                if ( ( ( int ) ( particles.level[part_id] ) <particles.nb_of_secondaries ) && secondary_part.E > 0. )
                {

                    int level = ( int ) ( particles.level[part_id] );
                    level = part_id * particles.nb_of_secondaries + level;
                    //                     printf ( "%d LEVEL %d id %d level %d \n",__LINE__,level,part_id, ( int ) ( particles.level[part_id] ) );
                    particles.sec_E[level] =  particles.E[part_id];
                    particles.sec_px[level] = particles.px[part_id];
                    particles.sec_py[level] = particles.py[part_id];
                    particles.sec_pz[level] = particles.pz[part_id];
                    particles.sec_dx[level] = particles.dx[part_id];
                    particles.sec_dy[level] = particles.dy[part_id];
                    particles.sec_dz[level] = particles.dz[part_id];
                    particles.sec_pname[level] = particles.pname[part_id];

                    particles.E[part_id]  = secondary_part.E;
                    particles.dx[part_id] = secondary_part.dir.x;
                    particles.dy[part_id] = secondary_part.dir.y;
                    particles.dz[part_id] = secondary_part.dir.z;
                    particles.pname[part_id] = secondary_part.pname;
                    secondaryParticleCreated = TRUE;

                    particles.level[part_id]+=1;

                }
                else
                {
                    dose_record_standard ( dosi, secondary_part.E, particles.px[part_id],particles.py[part_id],particles.pz[part_id] );
                }

                alongStepLength=0;
                freeLength=0.;


                totalLength+=trueStepLength;
                //                 Troncature(particles, id);
            } // significant_loss == false

        } // bool_loop == true

        if ( secondaryParticleCreated == TRUE ) return;
        //         break;




    }
    while ( ( particles.E[ part_id ] > EKINELIMIT ) && ( bool_loop ) );

    if ( ( particles.E[part_id]>EKINELIMIT ) /*&&(secondaryParticleCreated == FALSE)*/ ) //>1eV
    {

        ui8 next_discrete_process ;
        ui32 table_index; // indice de lecture de table de sections efficaces
        f32 next_interaction_distance = 1e9f;
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
        index_phantom.x = ui32 ( ( pos.x-vol.off_x ) * ivoxsize.x );
        index_phantom.y = ui32 ( ( pos.y-vol.off_y ) * ivoxsize.y );
        index_phantom.z = ui32 ( ( pos.z-vol.off_z ) * ivoxsize.z );
        index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                + index_phantom.y*vol.nb_vox_x
                + index_phantom.x; // linear index



        //Get mat index
        //         int mat = (int)(vol.data[index_phantom.w]);
        ui16 mat_id = vol.values[index_phantom.w];

        //// Get the next distance boundary volume /////////////////////////////////

        f32 vox_xmin = ((f32)index_phantom.x)*vol.spacing_x+vol.off_x;
        f32 vox_ymin = ((f32)index_phantom.y)*vol.spacing_y+vol.off_y;
        f32 vox_zmin = ((f32)index_phantom.z)*vol.spacing_z+vol.off_z;
        f32 vox_xmax = vox_xmin + vol.spacing_x;
        f32 vox_ymax = vox_ymin + vol.spacing_y;
        f32 vox_zmax = vox_zmin + vol.spacing_z;

        
        
        if( (index_phantom.x >= vol.nb_vox_x) ||
                (index_phantom.y >= vol.nb_vox_y) ||
                (index_phantom.z >= vol.nb_vox_z) )
        {
            particles.endsimu[part_id] = PARTICLE_FREEZE;
            return;
        }
        
        f32 fragment = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                      vox_ymin, vox_ymax, vox_zmin, vox_zmax );


        // Get distance to edge of voxel
        //         f32 fragment = get_boundary_voxel_by_raycasting(index_phantom, pos, direction, vol.voxel_size, part_id);
        fragment+=1.E-2*mm;
        //         printf("d_table.nb_bins %u\n",electron_CS_table.nb_bins);
        // Read Cross section table to get dedx, erange, lambda
        e_read_CS_table ( mat_id, energy, electron_CS_table, next_discrete_process, table_index, next_interaction_distance,
                          dedxeIoni,dedxeBrem,erange, lambda, randomnumbereBrem, randomnumbereIoni, parameters );
        //         printf("d_table.nb_bins %u\n",electron_CS_table.nb_bins);

        f32 cutstep = StepFunction ( erange );
        /*for(int i = 0;i<vol.number_of_voxels;i++){
            if(vol.values[i]!=0){ printf("%d %d %d %d\n",__LINE__,part_id,i,vol.values[i]); }
        }    */
        trueStepLength = GlobalMscScattering ( fragment, cutstep, erange, energy, lambda,   dedxeIoni,  dedxeBrem,  electron_CS_table,  mat_id, particles,  part_id, par1, par2, materials, dosi, index_phantom,vol,parameters );
        // for(int i = 0;i<vol.number_of_voxels;i++){
        //     if(vol.values[i]!=0){ printf("%d %d %d %d\n",__LINE__,part_id,i,vol.values[i]); }
        // }

        //         Troncature(particles, id);

        freeLength=alongStepLength+trueStepLength;

        totalLength+=trueStepLength;



    }
    else
    {
        particles.endsimu[part_id] = PARTICLE_DEAD;
        return;
    }


    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance (pos, vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax, EPSILON3 ) )
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
    index_phantom.x = ui32 ( ( pos.x-vol.off_x ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( pos.y-vol.off_y ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pos.z-vol.off_z ) * ivoxsize.z );

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

    f32 vox_xmin = index_phantom.x*vol.spacing_x+vol.off_x;
    f32 vox_ymin = index_phantom.y*vol.spacing_y+vol.off_y;
    f32 vox_zmin = index_phantom.z*vol.spacing_z+vol.off_z;
    f32 vox_xmax = vox_xmin + vol.spacing_x;
    f32 vox_ymax = vox_ymin + vol.spacing_y;
    f32 vox_zmax = vox_zmin + vol.spacing_z;

    // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
    // TODO: In theory this have to be applied just at the entry of the particle within the volume
    //       in order to avoid particle entry between voxels. Then, computing improvement can be made
    //       by calling this function only once, just for the particle step=0.    - JB
    pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                            vox_ymin, vox_ymax, vox_zmin, vox_zmax, EPSILON6 );

    f32 boundary_distance = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                           vox_ymin, vox_ymax, vox_zmin, vox_zmax );

    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + EPSILON3; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // get safety position (outside the current voxel)
    pos = transport_get_safety_outside_AABB( pos, vox_xmin, vox_xmax,
                                             vox_ymin, vox_ymax, vox_zmin, vox_zmax, EPSILON6 );

    // store new position
    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance (pos, vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax, EPSILON3 ) )
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

        /// If there is a secondary particle, push the primary into buffer and track this new particle

        if ( electron.endsimu == PARTICLE_ALIVE && electron.E > materials.electron_energy_cut[ mat_id ] &&
             particles.level[ part_id ] < particles.nb_of_secondaries )
        {

            // Get the absolute index into secondary buffer
            ui32 index_level = part_id * particles.nb_of_secondaries + ( ui32 ) particles.level[ part_id ];

            // Store the current particle
            particles.sec_E[ index_level ]  =  particles.E[ part_id ];
            particles.sec_px[ index_level ] = particles.px[ part_id ];
            particles.sec_py[ index_level ] = particles.py[ part_id ];
            particles.sec_pz[ index_level ] = particles.pz[ part_id ];
            particles.sec_dx[ index_level ] = particles.dx[ part_id ];
            particles.sec_dy[ index_level ] = particles.dy[ part_id ];
            particles.sec_dz[ index_level ] = particles.dz[ part_id ];
            particles.sec_pname[ index_level ] = particles.pname[ part_id ];

            // Fille the main buffer with the new secondary particle
            particles.E[ part_id ]  = electron.E;
            particles.dx[ part_id ] = electron.dir.x;
            particles.dy[ part_id ] = electron.dir.y;
            particles.dz[ part_id ] = electron.dir.z;
            particles.pname[ part_id ] = electron.pname;

            // Lose a level in the hierarchy
            particles.level[ part_id ] += 1;

        }
        else
        {
            // Drop energy if need
            if ( electron.E > 0.0 )
            {
               dose_record_standard( dosi, electron.E, particles.px[ part_id ],
                                     particles.py[ part_id ], particles.pz[ part_id ] );
            }
        }

    } // discrete process

    //// Photon energy cut
    if ( particles.E[ part_id ] <= materials.electron_energy_cut[ mat_id ] )
    {
        // Kill without mercy
        particles.endsimu[ part_id ] = PARTICLE_DEAD;
        // Drop energy
        dose_record_standard( dosi, particles.E[ part_id ], particles.px[ part_id ],
                              particles.py[ part_id ], particles.pz[ part_id ] );
    }

}

// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPDN::kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
        f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{  
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, EPSILON6, id);
}


// Host Kernel that move particles to the voxelized volume boundary
void VPDN::kernel_host_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 part_id )
{       
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, EPSILON6, part_id);
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
    f32 randomnumbereIoni= -logf ( JKISS32 ( particles, id ) ); // -log(RN)
    f32 randomnumbereBrem= -logf ( JKISS32 ( particles, id ) ); // -log(RN)
    f32 freeLength = 0.0*mm;

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {

        if ( particles.pname[id] == PHOTON )
        {
            VPDN::track_photon_to_out ( particles, vol, materials, photon_CS_table, parameters, dosi, id );

        }
        else if ( particles.pname[id] == ELECTRON )
        {
            VPDN::track_electron_to_out ( particles, vol, materials, electron_CS_table, parameters, dosi,
                                          randomnumbereIoni, randomnumbereBrem, freeLength, id );
        }

        // Condition if particle is dead and if it was a secondary
        if ( ( ( particles.endsimu[id]==PARTICLE_DEAD ) || ( particles.endsimu[id]==PARTICLE_FREEZE ) ) && ( particles.level[id]>PRIMARY ) )
        {

            /// Pull back the particle stored in the secondary buffer

            // Wake up the particle
            particles.endsimu[id] = PARTICLE_ALIVE;
            // Earn a higher level
            particles.level[id]  -= 1;
            // Get the absolute index into secondary buffer
            ui32 index_level = id * particles.nb_of_secondaries + ( ui32 ) particles.level[id];

            // FreeLength must be reinitialized due to voxels navigation (diff mats)
            freeLength = 0.0*mm;
            randomnumbereIoni= -logf ( JKISS32 ( particles, id ) ); // -log(RN)
            randomnumbereBrem= -logf ( JKISS32 ( particles, id ) ); // -log(RN)

            // Get back the stored particle into the primary buffer
            particles.E[ id ]     = particles.sec_E[ index_level ]    ;
            particles.px[ id ]    = particles.sec_px[ index_level ]   ;
            particles.py[ id ]    = particles.sec_py[ index_level ]   ;
            particles.pz[ id ]    = particles.sec_pz[ index_level ]   ;
            particles.dx[ id ]    = particles.sec_dx[ index_level ]   ;
            particles.dy[ id ]    = particles.sec_dy[ index_level ]   ;
            particles.dz[ id ]    = particles.sec_dz[ index_level ]   ;
            particles.pname[ id ] = particles.sec_pname[ index_level ];
        }
    }

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
    f32 randomnumbereIoni= -logf ( JKISS32 ( particles, id ) ); // -log(RN)
    f32 randomnumbereBrem= -logf ( JKISS32 ( particles, id ) ); // -log(RN)
    f32 freeLength = 0.0*mm;

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {

        if ( particles.pname[id] == PHOTON )
        {
            VPDN::track_photon_to_out ( particles, vol, materials, photon_CS_table, parameters, dosi, id );

        }
        else if ( particles.pname[id] == ELECTRON )
        {
            VPDN::track_electron_to_out ( particles, vol, materials, electron_CS_table, parameters, dosi,
                                          randomnumbereIoni, randomnumbereBrem, freeLength, id );
        }

        // Condition if particle is dead and if it was a secondary
        if ( ( ( particles.endsimu[id]==PARTICLE_DEAD ) || ( particles.endsimu[id]==PARTICLE_FREEZE ) ) && ( particles.level[id]>PRIMARY ) )
        {

            /// Pull back the particle stored in the secondary buffer

            // Wake up the particle
            particles.endsimu[id] = PARTICLE_ALIVE;
            // Earn a higher level
            particles.level[id]  -= 1;
            // Get the absolute index into secondary buffer
            ui32 index_level = id * particles.nb_of_secondaries + ( ui32 ) particles.level[id];

            // FreeLength must be reinitialized due to voxels navigation (diff mats)
            freeLength = 0.0*mm;
            randomnumbereIoni= -logf ( JKISS32 ( particles, id ) ); // -log(RN)
            randomnumbereBrem= -logf ( JKISS32 ( particles, id ) ); // -log(RN)

            // Get back the stored particle into the primary buffer
            particles.E[ id ]     = particles.sec_E[ index_level ]    ;
            particles.px[ id ]    = particles.sec_px[ index_level ]   ;
            particles.py[ id ]    = particles.sec_py[ index_level ]   ;
            particles.pz[ id ]    = particles.sec_pz[ index_level ]   ;
            particles.dx[ id ]    = particles.sec_dx[ index_level ]   ;
            particles.dy[ id ]    = particles.sec_dy[ index_level ]   ;
            particles.dz[ id ]    = particles.sec_dz[ index_level ]   ;
            particles.pname[ id ] = particles.sec_pname[ index_level ];
        }
    }
}

////:: Privates

bool VoxPhanDosiNav::m_check_mandatory()
{

    if ( m_phantom.data_h.nb_vox_x == 0 || m_phantom.data_h.nb_vox_y == 0 || m_phantom.data_h.nb_vox_z == 0 ||
            m_phantom.data_h.spacing_x == 0 || m_phantom.data_h.spacing_y == 0 || m_phantom.data_h.spacing_z == 0 ||
            m_phantom.list_of_materials.size() == 0 )
    {
        return false;
    }
    else
    {
        return true;
    }

}

////:: Main functions

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
                                                                               m_phantom.data_d.zmin, m_phantom.data_d.zmax );
        cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to in)" );
        cudaThreadSynchronize();
    }


}

void VoxPhanDosiNav::track_to_out ( Particles particles )
{
    //
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {

        ui32 id=0;
        while ( id<particles.size )
        {

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
    if( !m_elements_filename.empty() )
    {
        m_materials.load_elements_database( m_elements_filename );
    }
    else
    {
        m_materials.load_elements_database();
    }

    if( !m_materials_filename.empty() )
    {
        m_materials.load_materials_database( m_materials_filename );
    }
    else
    {
        m_materials.load_materials_database();
    }
    
    
    // Materials table
    //     m_materials.load_elements_database();
    //     m_materials.load_materials_database();
    m_materials.initialize ( m_phantom.list_of_materials, params );

    // Cross Sections
    m_cross_sections.initialize ( m_materials, params );

    // Init dose map
    m_dose_calculator.set_size_in_voxel ( m_phantom.data_h.nb_vox_x,
                                          m_phantom.data_h.nb_vox_y,
                                          m_phantom.data_h.nb_vox_z );
    m_dose_calculator.set_voxel_size ( m_phantom.data_h.spacing_x,
                                       m_phantom.data_h.spacing_y,
                                       m_phantom.data_h.spacing_z );
    m_dose_calculator.set_offset ( m_phantom.data_h.off_x,
                                   m_phantom.data_h.off_y,
                                   m_phantom.data_h.off_z );
    m_dose_calculator.initialize ( m_params ); // CPU&GPU
    
    m_dose_calculator.set_voxelized_phantom(m_phantom);
    m_dose_calculator.set_materials(m_materials);

}

void VoxPhanDosiNav::calculate_dose_to_water(){

    m_dose_calculator.calculate_dose_to_water();

}

void VoxPhanDosiNav::calculate_dose_to_phantom(){

    m_dose_calculator.calculate_dose_to_phantom();

}

void VoxPhanDosiNav::set_elements( std::string filename )
{
    m_elements_filename = filename;
}

void VoxPhanDosiNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}




#endif
