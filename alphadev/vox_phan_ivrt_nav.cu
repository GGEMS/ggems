// GGEMS Copyright (C) 2017

/*!
 * \file vox_phan_img_nav.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 18 novembre 2015
 *
 * v0.2: JB - Change all structs and remove CPU exec *
 */

#ifndef VOX_PHAN_IVRT_NAV_CU
#define VOX_PHAN_IVRT_NAV_CU

#include "vox_phan_ivrt_nav.cuh"

////:: GPU Codes

__device__ void VPIVRTN::track_to_out( ParticlesData *particles,
                                    const VoxVolumeData<ui16> *vol,
                                    const MaterialsData *materials,
                                    const PhotonCrossSectionData *photon_CS_table,
                                    const GlobalSimulationParametersData *parameters,
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
    index_phantom.x = ui32 ( ( pos.x+vol->off_x ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( pos.y+vol->off_y ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pos.z+vol->off_z ) * ivoxsize.z );

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
    f32 boundary_distance = hit_ray_AABB( pos, dir, vox_xmin, vox_xmax,
                                          vox_ymin, vox_ymax, vox_zmin, vox_zmax );

    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters->geom_tolerance; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // get safety position (outside the current voxel)
    pos = transport_get_safety_outside_AABB( pos, vox_xmin, vox_xmax,
                                             vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters->geom_tolerance );

    // update tof
    particles->tof[part_id] += c_light * next_interaction_distance;

    // store new position
    particles->px[part_id] = pos.x;
    particles->py[part_id] = pos.y;
    particles->pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance( pos, vol->xmin, vol->xmax, vol->ymin, vol->ymax,
                                          vol->zmin, vol->zmax, parameters->geom_tolerance ) )
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

        // If the process is PHOTON_COMPTON or PHOTON_RAYLEIGH the scatter
        // order is incremented
        if( next_discrete_process == PHOTON_COMPTON
                || next_discrete_process == PHOTON_RAYLEIGH )
        {
            particles->scatter_order[ part_id ] += 1;
        }

        //// Here e- are not tracked, and lost energy not drop
        //// Energy cut
        if ( particles->E[ part_id ] <= materials->photon_energy_cut[ mat_id ])
        {
            // kill without mercy (energy not drop)
            particles->status[part_id] = PARTICLE_DEAD;
            return;
        }
    }    
}

__device__ void VPIVRTN::track_to_out_woodcock( ParticlesData *particles,
                                                const VoxVolumeData<ui16> *vol,
                                                const MaterialsData *materials,
                                                const PhotonCrossSectionData *photon_CS_table,
                                                const GlobalSimulationParametersData *parameters,
                                                ui32 part_id,
                                                f32* mumax_table )
{
    // Read position
    f32xyz pos;
    pos.x = particles->px[part_id];
    pos.y = particles->py[part_id];
    pos.z = particles->pz[part_id];

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / vol->spacing_x;
    ivoxsize.y = 1.0 / vol->spacing_y;
    ivoxsize.z = 1.0 / vol->spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32( ( pos.x + vol->off_x ) * ivoxsize.x );
    index_phantom.y = ui32( ( pos.y + vol->off_y ) * ivoxsize.y );
    index_phantom.z = ui32( ( pos.z + vol->off_z ) * ivoxsize.z );

    index_phantom.w = index_phantom.z*vol->nb_vox_x*vol->nb_vox_y
            + index_phantom.y*vol->nb_vox_x
            + index_phantom.x; // linear index

    // Read direction
    f32xyz dir;
    dir.x = particles->dx[part_id];
    dir.y = particles->dy[part_id];
    dir.z = particles->dz[part_id];

    // Vars
    f32 next_interaction_distance;

    //// Find next discrete interaction ///////////////////////////////////////

    // Search the energy index to read CS
    f32 energy = particles->E[part_id];
    ui32 E_index = binary_search( energy, photon_CS_table->E_bins,
                                  photon_CS_table->nb_bins );

    // Get index CS table (considering mat id)
    f32 CS_max = get_CS_from_table( photon_CS_table->E_bins, mumax_table,
                                    energy, E_index, E_index );

    // Woodcock tracking
    next_interaction_distance = -log( prng_uniform( particles, part_id ) ) * CS_max;

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // updates
    particles->tof[part_id] += c_light * next_interaction_distance;
    particles->next_interaction_distance[part_id] = next_interaction_distance;

    // store the new position
    particles->px[part_id] = pos.x;
    particles->py[part_id] = pos.y;
    particles->pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance( pos, vol->xmin, vol->xmax, vol->ymin, vol->ymax,
                                          vol->zmin, vol->zmax, parameters->geom_tolerance ) )
    {
        particles->status[part_id] = PARTICLE_FREEZE;
        return;
    }

    //// Real or fictif process /////////////////////////////////////////////////

    // Defined index phantom
    index_phantom.x = ui32( ( pos.x + vol->off_x ) * ivoxsize.x );
    index_phantom.y = ui32( ( pos.y + vol->off_y ) * ivoxsize.y );
    index_phantom.z = ui32( ( pos.z + vol->off_z ) * ivoxsize.z );

    index_phantom.w = index_phantom.z*vol->nb_vox_x*vol->nb_vox_y
            + index_phantom.y*vol->nb_vox_x
            + index_phantom.x; // linear index

    // Get the material that compose this volume
    ui16 mat_id = vol->values[ index_phantom.w ];

    // Get index CS table (considering mat id)
    ui32 CS_index = mat_id*photon_CS_table->nb_bins + E_index;
    f32 sum_CS = 0.0;
    f32 CS_PE = 0.0;
    f32 CS_CPT = 0.0;
    f32 CS_RAY = 0.0;
    ui8 next_discrete_process = 0;
    f32 interaction_distance;
    next_interaction_distance = F32_MAX;

    if ( parameters->physics_list[PHOTON_PHOTOELECTRIC] )
    {
        CS_PE = get_CS_from_table( photon_CS_table->E_bins, photon_CS_table->Photoelectric_Std_CS,
                                   energy, E_index, CS_index );
        sum_CS += CS_PE;
    }

    if ( parameters->physics_list[PHOTON_COMPTON] )
    {
        CS_CPT = get_CS_from_table( photon_CS_table->E_bins, photon_CS_table->Compton_Std_CS,
                                    energy, E_index, CS_index );
        sum_CS += CS_CPT;
    }

    if ( parameters->physics_list[PHOTON_RAYLEIGH] )
    {
        CS_RAY = get_CS_from_table( photon_CS_table->E_bins, photon_CS_table->Rayleigh_Lv_CS,
                                    energy, E_index, CS_index );
        sum_CS += CS_RAY;
    }

    f32 rnd = prng_uniform( particles, part_id );

    if ( rnd > sum_CS * CS_max  )
    {
        // Fictive interaction, keep going!
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    // Resolve process
    if ( parameters->physics_list[PHOTON_PHOTOELECTRIC] )
    {
        rnd = prng_uniform( particles, part_id );
        interaction_distance = -log( rnd ) / CS_PE;
        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_PHOTOELECTRIC;
        }
    }

    if ( parameters->physics_list[PHOTON_COMPTON] )
    {
        rnd = prng_uniform( particles, part_id );
        interaction_distance = -log( rnd ) / CS_CPT;
        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_COMPTON;
        }
    }

    if ( parameters->physics_list[PHOTON_RAYLEIGH] )
    {
        rnd = prng_uniform( particles, part_id );
        interaction_distance = -log( rnd ) / CS_RAY;
        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_RAYLEIGH;
        }
    }

    // update
    particles->next_discrete_process[part_id] = next_discrete_process;

    // Resolve discrete process
    SecParticle electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                             materials, mat_id, part_id );

    // If the process is PHOTON_COMPTON or PHOTON_RAYLEIGH the scatter
    // order is incremented
    if( next_discrete_process == PHOTON_COMPTON
            || next_discrete_process == PHOTON_RAYLEIGH )
    {
        particles->scatter_order[ part_id ] += 1;
    }

    //// Here e- are not tracked, and lost energy not drop
    //// Energy cut
    if ( particles->E[ part_id ] <= materials->photon_energy_cut[ mat_id ])
    {
        // kill without mercy (energy not drop)
        particles->status[part_id] = PARTICLE_DEAD;
        return;
    }
}


//// Experimental Super Voxel Woodcock

__device__ void VPIVRTN::track_to_out_svw( ParticlesData *particles,
                                    const VoxVolumeData<ui16> *vol,
                                    const MaterialsData *materials,
                                    const PhotonCrossSectionData *photon_CS_table,
                                    const GlobalSimulationParametersData *parameters,
                                    ui32 part_id,
                                    f32* mumax_table,
                                    ui16* mumax_index_table,
                                    ui32 nb_bins_sup_voxel )
{
    f32 sv_spacing_x = nb_bins_sup_voxel * vol->spacing_x;
    f32 sv_spacing_y = nb_bins_sup_voxel * vol->spacing_y;
    f32 sv_spacing_z = nb_bins_sup_voxel * vol->spacing_z;

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

    // Vars
    f32 next_interaction_distance;

    //// Find next discrete interaction ///////////////////////////////////////

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / vol->spacing_x;
    ivoxsize.y = 1.0 / vol->spacing_y;
    ivoxsize.z = 1.0 / vol->spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32( ( pos.x + vol->off_x ) * ivoxsize.x );
    index_phantom.y = ui32( ( pos.y + vol->off_y ) * ivoxsize.y );
    index_phantom.z = ui32( ( pos.z + vol->off_z ) * ivoxsize.z );

    index_phantom.w = index_phantom.z*vol->nb_vox_x*vol->nb_vox_y
                        + index_phantom.y*vol->nb_vox_x
                        + index_phantom.x; // linear index

    // Search the energy index to read CS
    f32 energy = particles->E[part_id];
    ui32 E_index = binary_search( energy, photon_CS_table->E_bins,
                                  photon_CS_table->nb_bins );

    // Get index CS table the coresponding super voxel
    ui32 CS_max_index = mumax_index_table[ index_phantom.w * photon_CS_table->nb_bins + E_index ] * photon_CS_table->nb_bins + E_index;

    f32 CS_max = ( E_index == 0 )? mumax_table[CS_max_index]: linear_interpolation(photon_CS_table->E_bins[E_index-1], mumax_table[CS_max_index-1],
            photon_CS_table->E_bins[E_index], mumax_table[CS_max_index], energy);//*******//

    // Woodcock tracking
    next_interaction_distance = -log( prng_uniform( particles, part_id ) ) * CS_max;
    //interaction_distance  = next_interaction_distance;

    //// Get the next distance boundary volume /////////////////////////////////

    ui32 sv_index_phantom_x = index_phantom.x / nb_bins_sup_voxel;
    ui32 sv_index_phantom_y = index_phantom.y / nb_bins_sup_voxel;
    ui32 sv_index_phantom_z = index_phantom.z / nb_bins_sup_voxel;

    f32 sv_vox_xmin = sv_index_phantom_x * sv_spacing_x - vol->off_x;
    f32 sv_vox_ymin = sv_index_phantom_y * sv_spacing_y - vol->off_y;
    f32 sv_vox_zmin = sv_index_phantom_z * sv_spacing_z - vol->off_z;
    f32 sv_vox_xmax = sv_vox_xmin + sv_spacing_x;
    f32 sv_vox_ymax = sv_vox_ymin + sv_spacing_y;
    f32 sv_vox_zmax = sv_vox_zmin + sv_spacing_z;

    // get a safety position for the particle within this super voxel (sometime a particle can be right between two super voxels)

    pos = transport_get_safety_inside_AABB( pos, sv_vox_xmin, sv_vox_xmax,
                                            sv_vox_ymin, sv_vox_ymax, sv_vox_zmin, sv_vox_zmax, parameters->geom_tolerance );

    f32 boundary_distance = hit_ray_AABB( pos, dir, sv_vox_xmin, sv_vox_xmax,
                                          sv_vox_ymin, sv_vox_ymax, sv_vox_zmin, sv_vox_zmax );

    //// Move particle //////////////////////////////////////////////////////

    ui8 next_discrete_process = 0;
    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters->geom_tolerance; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;

    }

    // get the new position
    pos = fxyz_add( pos, fxyz_scale( dir, next_interaction_distance ) );

    // updates
    particles->tof[part_id] += c_light * next_interaction_distance;
    particles->next_interaction_distance[part_id] = next_interaction_distance;
    particles->next_discrete_process[part_id] = next_discrete_process;//*******//

    // store the new position
    particles->px[part_id] = pos.x;
    particles->py[part_id] = pos.y;
    particles->pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance ( pos, vol->xmin, vol->xmax, vol->ymin, vol->ymax,
                                           vol->zmin, vol->zmax, parameters->geom_tolerance ) )
    {
        particles->status[part_id] = PARTICLE_FREEZE;
        return;
    }

    if ( next_discrete_process != GEOMETRY_BOUNDARY )
    {

        // Get the material that compose this volume
        index_phantom.x = ui32( ( pos.x + vol->off_x ) * ivoxsize.x );
        index_phantom.y = ui32( ( pos.y + vol->off_y ) * ivoxsize.y );
        index_phantom.z = ui32( ( pos.z + vol->off_z ) * ivoxsize.z );

        index_phantom.w = index_phantom.z*vol->nb_vox_x*vol->nb_vox_y
                + index_phantom.y*vol->nb_vox_x
                + index_phantom.x; // linear index

        ui16 mat_id = vol->values[ index_phantom.w ];


        //// Choose real or fictitious process ///////////////////////////////////////

        // Get index CS table (considering mat id)
        ui32 CS_index = mat_id*photon_CS_table->nb_bins + E_index;
        f32 sum_CS = 0.0;
        f32 CS_PE = 0.0;
        f32 CS_CPT = 0.0;
        f32 CS_RAY = 0.0;
        next_discrete_process = 0;
        f32 interaction_distance;
        next_interaction_distance = F32_MAX;

        if ( parameters->physics_list[PHOTON_PHOTOELECTRIC] )
        {
            CS_PE = get_CS_from_table( photon_CS_table->E_bins, photon_CS_table->Photoelectric_Std_CS,
                                       energy, E_index, CS_index );
            sum_CS += CS_PE;
        }

        if ( parameters->physics_list[PHOTON_COMPTON] )
        {
            CS_CPT = get_CS_from_table( photon_CS_table->E_bins, photon_CS_table->Compton_Std_CS,
                                        energy, E_index, CS_index );
            sum_CS += CS_CPT;
        }

        if ( parameters->physics_list[PHOTON_RAYLEIGH] )
        {
            CS_RAY = get_CS_from_table( photon_CS_table->E_bins, photon_CS_table->Rayleigh_Lv_CS,
                                        energy, E_index, CS_index );
            sum_CS += CS_RAY;
        }

        f32 rnd = prng_uniform( particles, part_id );

        if ( rnd > sum_CS * CS_max )
        {
            // Fictive interaction
            return;
        }

/*
        // Resolve discrete process
        SecParticle electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                                 materials, mat_id, part_id );
        //// Here e- are not tracked, and lost energy not drop
        //// Energy cut
        if ( particles->E[ part_id ] <= materials->photon_energy_cut[ mat_id ])
        {
            // kill without mercy (energy not drop)
            particles->status[part_id] = PARTICLE_DEAD;
            return;
        }
*/

        //// Apply discrete process //////////////////////////////////////////////////

        // Resolve process
        if ( parameters->physics_list[PHOTON_PHOTOELECTRIC] )
        {
            rnd = prng_uniform( particles, part_id );
            interaction_distance = -log( rnd ) / CS_PE;
            if ( interaction_distance < next_interaction_distance )
            {
                next_interaction_distance = interaction_distance;
                next_discrete_process = PHOTON_PHOTOELECTRIC;
            }
        }

        if ( parameters->physics_list[PHOTON_COMPTON] )
        {
            rnd = prng_uniform( particles, part_id );
            interaction_distance = -log( rnd ) / CS_CPT;
            if ( interaction_distance < next_interaction_distance )
            {
                next_interaction_distance = interaction_distance;
                next_discrete_process = PHOTON_COMPTON;
            }
        }

        if ( parameters->physics_list[PHOTON_RAYLEIGH] )
        {
            rnd = prng_uniform( particles, part_id );
            interaction_distance = -log( rnd ) / CS_RAY;
            if ( interaction_distance < next_interaction_distance )
            {
                next_interaction_distance = interaction_distance;
                next_discrete_process = PHOTON_RAYLEIGH;
            }
        }

        particles->next_discrete_process[part_id] = next_discrete_process;

        // Resolve discrete process
        /*SecParticle electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                                 materials, mat_id, part_id );*/

        SecParticle electron;
        electron.endsimu = PARTICLE_DEAD;
        electron.dir.x = 0.;
        electron.dir.y = 0.;
        electron.dir.z = 1.;
        electron.E = 0.;

        if ( next_discrete_process == PHOTON_COMPTON )
        {
            electron = Compton_SampleSecondaries_standard( particles, materials->electron_energy_cut[mat_id],
                       parameters->secondaries_list[ELECTRON], part_id );
        }

        if ( next_discrete_process == PHOTON_PHOTOELECTRIC )
        {
            electron = Photoelec_SampleSecondaries_standard( particles, materials, photon_CS_table,
                       particles->E_index[part_id], materials->electron_energy_cut[mat_id],
                       mat_id, parameters->secondaries_list[ELECTRON], part_id );
        }

        if ( next_discrete_process == PHOTON_RAYLEIGH )
        {
            Rayleigh_SampleSecondaries_Livermore( particles, materials, photon_CS_table, particles->E_index[part_id],
                                                  mat_id, part_id );
        }

        //// Here e- are not tracked, and lost energy not drop
        //// Energy cut

        // If the process is PHOTON_COMPTON or PHOTON_RAYLEIGH the scatter
        // order is incremented
        if( next_discrete_process == PHOTON_COMPTON
                || next_discrete_process == PHOTON_RAYLEIGH )
        {
            particles->scatter_order[ part_id ] += 1;
        }

        if ( particles->E[ part_id ] <= materials->photon_energy_cut[ mat_id ])
        {
            // kill without mercy (energy not drop)
            particles->status[part_id] = PARTICLE_DEAD;
            return;
        }
    }
}

// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPIVRTN::kernel_device_track_to_in ( ParticlesData *particles, f32 xmin, f32 xmax,
                                                  f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 geom_tolerance )
{

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles->size ) return;

    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, geom_tolerance, id );
}

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPIVRTN::kernel_device_track_to_out ( ParticlesData *particles,
                                                   const VoxVolumeData<ui16> *vol,
                                                   const MaterialsData *materials,
                                                   const PhotonCrossSectionData *photon_CS_table,
                                                   const GlobalSimulationParametersData *parameters )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles->size ) return;

    while ( particles->status[id] != PARTICLE_DEAD && particles->status[id] != PARTICLE_FREEZE )
    {        
        VPIVRTN::track_to_out( particles, vol, materials, photon_CS_table, parameters, id );
    }
}

// Device kernel that track particles within the voxelized volume until boundary with woodcock tracking
__global__ void VPIVRTN::kernel_device_track_to_out_woodcock ( ParticlesData *particles,
                                                               const VoxVolumeData<ui16> *vol,
                                                               const MaterialsData *materials,
                                                               const PhotonCrossSectionData *photon_CS_table,
                                                               const GlobalSimulationParametersData *parameters,
                                                               f32* mumax_table )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles->size ) return;

    while ( particles->status[id] != PARTICLE_DEAD && particles->status[id] != PARTICLE_FREEZE )
    {
        VPIVRTN::track_to_out_woodcock( particles, vol, materials, photon_CS_table, parameters, id, mumax_table );
    }
}

// Device kernel that track particles within the voxelized volume until super voxel boundary with woodcock tracking
__global__ void VPIVRTN::kernel_device_track_to_out_svw ( ParticlesData *particles,
                                                   const VoxVolumeData<ui16> *vol,
                                                   const MaterialsData *materials,
                                                   const PhotonCrossSectionData *photon_CS_table,
                                                   const GlobalSimulationParametersData *parameters,
                                                   f32* mumax_table,
                                                   ui16 *mumax_index_table,
                                                   ui32 nb_bins_sup_voxel )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles->size ) return;

    while ( particles->status[id] != PARTICLE_DEAD && particles->status[id] != PARTICLE_FREEZE )
    {
        VPIVRTN::track_to_out_svw( particles, vol, materials, photon_CS_table, parameters, id, mumax_table,
                                mumax_index_table, nb_bins_sup_voxel );
    }
}

////:: Privates

bool VoxPhanIVRTNav::m_check_mandatory()
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
ui64 VoxPhanIVRTNav::m_get_memory_usage()
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

    // If Super Voxel Woodcock
    if ( m_flag_vrt == IMG_VRT_WOODCOCK || m_flag_vrt == IMG_VRT_SVW )
    {
        mem += m_cross_sections.h_photon_CS->nb_bins * sizeof(ui32);
    }

    return mem;
}

// Use for woodcock navigation
void VoxPhanIVRTNav::m_build_mumax_table()
{
    // Init mumax table vector
    ui32 nb_bins_E = m_cross_sections.h_photon_CS->nb_bins;
    HANDLE_ERROR( cudaMallocManaged( &(m_mumax_table), nb_bins_E * sizeof( ui32 ) ) );

    // Find the most attenuate material
    f32 max_dens = 0.0;
    ui32 ind_mat = 0;
    ui32 i = 0; while ( i < m_materials.h_materials->nb_materials )
    {
        if ( m_materials.h_materials->density[i] > max_dens )
        {
            max_dens = m_materials.h_materials->density[ i ];
            ind_mat = i;
        }
        ++i;
    }

    // Build table using max density  [ 1 / Sum( CS ) ]
    i=0; while ( i < nb_bins_E )
    {
        ui32 index = ind_mat * nb_bins_E + i;
        f32 sum_CS = 0.0;

        if ( mh_params->physics_list[PHOTON_PHOTOELECTRIC] )
        {
            sum_CS += m_cross_sections.h_photon_CS->Photoelectric_Std_CS[ index ];
        }

        if ( mh_params->physics_list[PHOTON_COMPTON] )
        {
            sum_CS += m_cross_sections.h_photon_CS->Compton_Std_CS[ index ];
        }

        if ( mh_params->physics_list[PHOTON_RAYLEIGH] )
        {
            sum_CS += m_cross_sections.h_photon_CS->Rayleigh_Lv_CS[ index ];
        }

        m_mumax_table[ i ] = 1.0 / sum_CS;
        ++i;
    }
}

// Use for super voxel woodcock navigation
void VoxPhanIVRTNav::m_build_svw_mumax_table()
{
    ui32 nb_bins_E = m_cross_sections.h_photon_CS->nb_bins;
    // Init voxel -> super voxel index
    ui32 *sup_vox_index = new ui32[m_phantom.h_volume->number_of_voxels];

    // Init the super voxel size
    ui32 nbx_sup_vox = (m_phantom.h_volume->nb_vox_x % m_nb_bins_sup_voxel == 0)
            ? m_phantom.h_volume->nb_vox_x / m_nb_bins_sup_voxel
            : m_phantom.h_volume->nb_vox_x / m_nb_bins_sup_voxel + 1;
    ui32 nby_sup_vox = (m_phantom.h_volume->nb_vox_y % m_nb_bins_sup_voxel == 0)
            ? m_phantom.h_volume->nb_vox_y / m_nb_bins_sup_voxel
            : m_phantom.h_volume->nb_vox_y / m_nb_bins_sup_voxel + 1;
    ui32 nbz_sup_vox = (m_phantom.h_volume->nb_vox_z % m_nb_bins_sup_voxel == 0)
            ? m_phantom.h_volume->nb_vox_z / m_nb_bins_sup_voxel
            : m_phantom.h_volume->nb_vox_z / m_nb_bins_sup_voxel + 1;

    // Init material mumax table
    f32 *mu_mat = new f32[ m_materials.h_materials->nb_materials  * nb_bins_E ];
    ui32 ind_mat = 0; while ( ind_mat < m_materials.h_materials->nb_materials )
    {
        ui32 n=0; while ( n < nb_bins_E )
        {
            ui32 index = ind_mat * nb_bins_E + n;
            f32 cs = 0.0;
            if ( mh_params->physics_list[PHOTON_PHOTOELECTRIC] )
            {
                cs += m_cross_sections.h_photon_CS->Photoelectric_Std_CS[ index ];
            }

            if ( mh_params->physics_list[PHOTON_COMPTON] )
            {
                cs += m_cross_sections.h_photon_CS->Compton_Std_CS[ index ];
            }

            if ( mh_params->physics_list[PHOTON_RAYLEIGH] )
            {
                cs += m_cross_sections.h_photon_CS->Rayleigh_Lv_CS[ index ];
            }
            mu_mat [ index ] = cs;
            ++n;
        }
        ++ind_mat;
    }


    // Find the less attenuate material
    ui16 *mumin_ind_mat = new ui16[ nb_bins_E ];
    ui16 ind_bins_E = 0; while ( ind_bins_E < nb_bins_E )
    {
        mumin_ind_mat [ ind_bins_E ] = 0;
        ind_mat = 1; while ( ind_mat < m_materials.h_materials->nb_materials )
        {
            if ( mu_mat [ ind_mat * nb_bins_E + ind_bins_E ] < mu_mat [ mumin_ind_mat[ ind_bins_E ] * nb_bins_E + ind_bins_E ] ) {
                mumin_ind_mat [ ind_bins_E ] = ind_mat;
            }
            ++ind_mat;
        }
        ++ind_bins_E;
    }

    // Init super voxels mumax table vector and material index
    ui16 *sup_vox_ind_mat_table = new ui16[ nbx_sup_vox * nby_sup_vox * nbz_sup_vox * nb_bins_E];
    ui32 ind_sup_vol = 0; while ( ind_sup_vol < nbx_sup_vox * nby_sup_vox * nbz_sup_vox )
    {
        ind_bins_E = 0; while ( ind_bins_E < nb_bins_E ) {
            sup_vox_ind_mat_table [ ind_sup_vol * nb_bins_E + ind_bins_E ] = mumin_ind_mat [ ind_bins_E ];
            ++ind_bins_E;
        }
        ++ind_sup_vol;
    }

    // Find the most attenuate material in each super voxel

    ui32 i, j, k, ii, jj, kk, rest;
    ui32 xy = m_phantom.h_volume->nb_vox_x * m_phantom.h_volume->nb_vox_y;
    ui32 sv_xy = nbx_sup_vox * nby_sup_vox;
    ui32 ind_vol = 0; while ( ind_vol < m_phantom.h_volume->number_of_voxels )
    {
        // Calculate the i, j, k voxel index
        k = ind_vol / xy;
        rest = ind_vol % xy;
        j = rest / m_phantom.h_volume->nb_vox_x;
        i = rest % m_phantom.h_volume->nb_vox_x;
        // Calculate the ii, jj, kk super voxel index
        ii = i / m_nb_bins_sup_voxel;
        jj = j / m_nb_bins_sup_voxel;
        kk = k / m_nb_bins_sup_voxel;
        ind_sup_vol = kk * sv_xy + jj * nbx_sup_vox + ii;

        // super voxel index associated to the the voxel ind_vol
        sup_vox_index[ ind_vol ] = ind_sup_vol;

        ind_mat = m_phantom.h_volume->values[ ind_vol ];

        // Material index associated to the super voxel ind_sup_vol according to E_bins
        ind_bins_E = 0; while ( ind_bins_E < nb_bins_E )
        {
            if ( mu_mat[ sup_vox_ind_mat_table [ ind_sup_vol * nb_bins_E + ind_bins_E ] * nb_bins_E + ind_bins_E ] < mu_mat[ ind_mat  * nb_bins_E + ind_bins_E ] )
            {
                sup_vox_ind_mat_table [ ind_sup_vol * nb_bins_E + ind_bins_E ] = ind_mat;
            }
            ++ind_bins_E;
        }
        ++ind_vol;
    }
/*  // debug test
    // Create an IO object of super voxels
        ImageIO *im_io = new ImageIO;

        std::string filename = "super_voxels_map.mhd";

        f32xyz offset;
        offset.x = m_phantom.h_volume->off_x;
        offset.y = m_phantom.h_volume->off_y;
        offset.z = m_phantom.h_volume->off_z;
        f32xyz spacing;
        spacing.x = m_phantom.h_volume->spacing_x;
        spacing.y = m_phantom.h_volume->spacing_y;
        spacing.z = m_phantom.h_volume->spacing_z;
        ui32xyz vol_size;
        vol_size.x = m_phantom.h_volume->nb_vox_x;
        vol_size.y = m_phantom.h_volume->nb_vox_y;
        vol_size.z = m_phantom.h_volume->nb_vox_z;

        im_io->write_3D(filename, sup_vox_index, vol_size, offset, spacing, false );

        // Create an IO object of super voxels max density
        f32 *tmp_density = new f32[m_phantom.h_volume->number_of_voxels];
        i = 0; while (i < m_phantom.h_volume->number_of_voxels) {
            tmp_density [ i ] = sup_vox_ind_mat_table [sup_vox_index[ i ] * nb_bins_E + 100 ];
            ++i;
        }
        im_io = new ImageIO;

        filename = "max_density_map_E100.mhd";

        im_io->write_3D( filename, tmp_density, vol_size, offset, spacing, false );
*/
    // Reduce the sup_vox_ind_mat table size (removing duplicates)

    std::vector<ui16> red_sup_vox_ind_mat_table(1, sup_vox_ind_mat_table [ 0 ]);
    ui16 *old_to_red_link = new ui16[ nbx_sup_vox * nby_sup_vox * nbz_sup_vox * nb_bins_E ];
    bool ind_not_found;
    old_to_red_link [0] = 0;
    ui32 ind = 1; while ( ind < nbx_sup_vox * nby_sup_vox * nbz_sup_vox * nb_bins_E)
    {
        ind_not_found = true;
        ui16 j = 0; while (j < red_sup_vox_ind_mat_table.size())
        {
            if ( sup_vox_ind_mat_table [ ind ] == red_sup_vox_ind_mat_table [ j ] )
            {
                old_to_red_link [ind] = j;
                ind_not_found = false;
                break;
            }
            ++j;
        }
        if (ind_not_found)
        {
            red_sup_vox_ind_mat_table.push_back(sup_vox_ind_mat_table [ ind ]);
            old_to_red_link [ ind ] = red_sup_vox_ind_mat_table.size() - 1;
        }
        ++ind;
    }

    // Link voxels to the reduced mumax index table
    HANDLE_ERROR( cudaMallocManaged( &(m_mumax_index_table), m_phantom.h_volume->number_of_voxels * nb_bins_E * sizeof( ui16 ) ) );
    ind_vol = 0; while ( ind_vol < m_phantom.h_volume->number_of_voxels )
    {
        ind_bins_E = 0; while ( ind_bins_E < nb_bins_E ) {
            m_mumax_index_table[ ind_vol * nb_bins_E + ind_bins_E ] = old_to_red_link[ sup_vox_index[ ind_vol ]  * nb_bins_E + ind_bins_E ];
            ++ind_bins_E;
        }
        ++ind_vol;
    }

    // Build table using max density  [ 1 / Sum( CS ) ]
    // Init voxels mumax table vector

    ui32 size = red_sup_vox_ind_mat_table.size() * nb_bins_E;
    HANDLE_ERROR( cudaMallocManaged( &(m_mumax_table), size * sizeof( f32 ) ) );

    ind_mat = 0; while ( ind_mat < red_sup_vox_ind_mat_table.size() )
    {
        ui32 j=0; while ( j < nb_bins_E )
        {
            ui32 index = red_sup_vox_ind_mat_table[ ind_mat ] * nb_bins_E + j;
            f32 sum_CS = 0.0;

            if ( mh_params->physics_list[PHOTON_PHOTOELECTRIC] )
            {
                sum_CS += m_cross_sections.h_photon_CS->Photoelectric_Std_CS[ index ];
            }

            if ( mh_params->physics_list[PHOTON_COMPTON] )
            {
                sum_CS += m_cross_sections.h_photon_CS->Compton_Std_CS[ index ];
            }

            if ( mh_params->physics_list[PHOTON_RAYLEIGH] )
            {
                sum_CS += m_cross_sections.h_photon_CS->Rayleigh_Lv_CS[ index ];
            }

            m_mumax_table[ ind_mat * nb_bins_E + j ] = 1.0 / sum_CS;
            ++j;
        }
        ++ind_mat;
    }
}


////:: Main functions

VoxPhanIVRTNav::VoxPhanIVRTNav()
{
    m_materials_filename = "";
    set_name( "VoxPhanIVRTNav" );

    // experimental (Super Voxel Woodcock tracking)
    m_mumax_table = nullptr;
    m_mumax_index_table = nullptr;
}

void VoxPhanIVRTNav::track_to_in(ParticlesData *d_particles )
{        
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 ) / mh_params->gpu_block_size;

    VPIVRTN::kernel_device_track_to_in<<<grid, threads>>>( d_particles, m_phantom.h_volume->xmin, m_phantom.h_volume->xmax,
                                                        m_phantom.h_volume->ymin, m_phantom.h_volume->ymax,
                                                        m_phantom.h_volume->zmin, m_phantom.h_volume->zmax,
                                                        mh_params->geom_tolerance );
    cudaDeviceSynchronize();
    cuda_error_check ( "Error ", " Kernel_VoxPhanIVRTNav (track to in)" );

}

void VoxPhanIVRTNav::track_to_out(ParticlesData *d_particles )
{
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 ) / mh_params->gpu_block_size;

    // DEBUG
    if ( m_flag_vrt == IMG_VRT_ANALOG )
    {
        VPIVRTN::kernel_device_track_to_out<<<grid, threads>>> ( d_particles, m_phantom.d_volume, m_materials.d_materials,
                                                              m_cross_sections.d_photon_CS, md_params );
    }
    else if ( m_flag_vrt == IMG_VRT_WOODCOCK )
    {
        VPIVRTN::kernel_device_track_to_out_woodcock<<<grid, threads>>> ( d_particles, m_phantom.d_volume, m_materials.d_materials,
                                                                  m_cross_sections.d_photon_CS, md_params, m_mumax_table );
    }
    else if ( m_flag_vrt == IMG_VRT_SVW )
    {
        VPIVRTN::kernel_device_track_to_out_svw<<<grid, threads>>> ( d_particles, m_phantom.d_volume, m_materials.d_materials,
                                                                  m_cross_sections.d_photon_CS, md_params, m_mumax_table,
                                                                  m_mumax_index_table, m_nb_bins_sup_voxel );
    }

    cudaDeviceSynchronize();
    cuda_error_check ( "Error ", " Kernel_VoxPhanIVRTNav (track to out)" );

}

void VoxPhanIVRTNav::load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
{   
    m_phantom.load_from_mhd( filename, range_mat_name );
}

void VoxPhanIVRTNav::initialize(GlobalSimulationParametersData *h_params, GlobalSimulationParametersData *d_params)
{
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanIVRTNav: missing parameters." );
        exit_simulation();
    }

    // Params
    mh_params = h_params;
    md_params = d_params;    

    // Phantom
    m_phantom.set_name( "VoxPhanIVRTNav" );
    m_phantom.initialize();

    // Material
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize( m_phantom.list_of_materials, h_params );

    // Cross Sections
    m_cross_sections.initialize( m_materials.h_materials, h_params );

    // If Woodcock init mumax table
    if ( m_flag_vrt == IMG_VRT_WOODCOCK )
    {
        m_build_mumax_table();
    }

    // If Super Voxel Woodcock init svw mumax table
    if ( m_flag_vrt == IMG_VRT_SVW )
    {
        m_build_svw_mumax_table();
    }

    // Some verbose if required
    if ( h_params->display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanIVRTNav", mem);
    }

}

void VoxPhanIVRTNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}

void VoxPhanIVRTNav::set_vrt( std::string kind )
{
    // Transform the name of the process in small letter
    std::transform( kind.begin(), kind.end(), kind.begin(), ::tolower );

    if ( kind == "analog" )
    {
        m_flag_vrt = IMG_VRT_ANALOG;
    }
    else if ( kind == "woodcock" )
    {
        m_flag_vrt = IMG_VRT_WOODCOCK;
    }
    else if ( kind == "svw" )
    {
        m_flag_vrt = IMG_VRT_SVW;
    }
    else
    {
        GGcerr << "Variance reduction technique not recognized: '" << kind << "'!" << GGendl;
        exit_simulation();
    }
}

// Set the super voxel size
void VoxPhanIVRTNav::set_nb_bins_sup_voxel( ui32 nb_bins_sup_voxel )
{
    m_nb_bins_sup_voxel = nb_bins_sup_voxel;
}

AabbData VoxPhanIVRTNav::get_bounding_box()
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
