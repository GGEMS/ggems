// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_iort_nav.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 23/03/2016
 *
 *
 *
 */

#ifndef VOX_PHAN_IORT_NAV_CU
#define VOX_PHAN_IORT_NAV_CU

#include "vox_phan_iort_nav.cuh"

////// HOST-DEVICE GPU Codes ////////////////////////////////////////////

__host__ __device__ void VPIORTN::track_to_out( ParticlesData &particles,
                                                VoxVolumeData<ui16> vol,
                                                MaterialsTable materials,
                                                PhotonCrossSectionTable photon_CS_table,
                                                GlobalSimulationParametersData parameters,
                                                DoseData dosi,
                                                Mu_MuEn_Table mu_table,
                                                HistoryMap hist_map,
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

    // Vars
    f32 next_interaction_distance;
    ui8 next_discrete_process;

    //// Find next discrete interaction ///////////////////////////////////////

#ifdef SKIP_VOXEL
    if ( mat_id != particles.geometry_id[ part_id ] )
    {
        // Store the material id
        particles.geometry_id[ part_id ] = mat_id;

        // Get next interaction and update values
        photon_get_next_interaction ( particles, parameters, photon_CS_table, mat_id, part_id );
    }

    // Read values
    next_interaction_distance = particles.next_interaction_distance[part_id];
    next_discrete_process = particles.next_discrete_process[part_id];

#else
    photon_get_next_interaction ( particles, parameters, photon_CS_table, mat_id, part_id );

    next_interaction_distance = particles.next_interaction_distance[part_id];
    next_discrete_process = particles.next_discrete_process[part_id];

#endif

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

        #ifdef SKIP_VOXEL
            // Update the the next distance interaction based on the length particle step
            particles.next_interaction_distance[ part_id ] -= ( boundary_distance + parameters.geom_tolerance );
        #endif

    }

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // get safety position (outside the current voxel)
    pos = transport_get_safety_outside_AABB( pos, vox_xmin, vox_xmax,
                                             vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters.geom_tolerance );

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance (pos, vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax, parameters.geom_tolerance ) )
    {
        particles.endsimu[part_id] = PARTICLE_FREEZE;
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    f32 energy = particles.E[ part_id ];

    // If TLE
    if ( mu_table.flag == TLE ) {

        if ( next_discrete_process != GEOMETRY_BOUNDARY )
        {
            // Resolve discrete process
            SecParticle electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                                     materials, mat_id, part_id );

            #ifdef SKIP_VOXEL
                particles.geometry_id[ part_id ] = 999999999; // Init with unknow material id to force process update
            #endif

        } // discrete process

        /// Drop energy ////////////

        // Get the mu_en for the current E
        ui32 E_index = binary_search ( energy, mu_table.E_bins, mu_table.nb_bins );

        f32 mu_en;

        if ( E_index == 0 )
        {
            mu_en = mu_table.mu_en[ mat_id*mu_table.nb_bins ];
        }
        else
        {
            mu_en = linear_interpolation( mu_table.E_bins[E_index-1],  mu_table.mu_en[mat_id*mu_table.nb_bins + E_index-1],
                                          mu_table.E_bins[E_index],    mu_table.mu_en[mat_id*mu_table.nb_bins + E_index],
                                          energy );
        }

        //                             record to the old position (current voxel)
        dose_record_TLE( dosi, energy, particles.px[ part_id ], particles.py[ part_id ],
                         particles.pz[ part_id ], next_interaction_distance,  mu_en );

        /// Energy cut /////////////

        // If gamma particle not enough energy (Energy cut)
        if ( particles.E[ part_id ] <= materials.photon_energy_cut[ mat_id ] )
        {
            // Kill without mercy
            particles.endsimu[ part_id ] = PARTICLE_DEAD;
        }

    }
    else // Else Analog or seTLE
    {

        // Resolve process
        SecParticle electron;
        if ( next_discrete_process != GEOMETRY_BOUNDARY )
        {
            // Resolve discrete process
            electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                                     materials, mat_id, part_id );

            #ifdef SKIP_VOXEL
                particles.geometry_id[ part_id ] = 999999999; // Init with unknow material id to force process update
            #endif

            /// Energy cut /////////////

            // If gamma particle not enough energy (Energy cut)
            if ( particles.E[ part_id ] <= materials.photon_energy_cut[ mat_id ] )
            {
                // Kill without mercy
                particles.endsimu[ part_id ] = PARTICLE_DEAD;
            }
        }

        // seTLE
        if ( mu_table.flag == seTLE )
        {
            /// seTLE /////////////////

            if ( next_discrete_process == PHOTON_COMPTON || next_discrete_process == PHOTON_RAYLEIGH )
            {
                // Increment the number of interactions at this position
                index_phantom.x = ui32 ( ( pos.x + vol.off_x ) * ivoxsize.x );
                index_phantom.y = ui32 ( ( pos.y + vol.off_y ) * ivoxsize.y );
                index_phantom.z = ui32 ( ( pos.z + vol.off_z ) * ivoxsize.z );

                index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                        + index_phantom.y*vol.nb_vox_x
                        + index_phantom.x; // linear index

                ggems_atomic_add( hist_map.interaction, index_phantom.w, ui32 ( 1 ) );

                // Add the particle energy in memory
                ggems_atomic_add( hist_map.energy, index_phantom.w, particles.E[ part_id ] );
            }

        }
        else // Analog
        {
            if ( next_discrete_process != GEOMETRY_BOUNDARY )
            {
                /// Drop energy ////////////

                // If gamma particle is dead (PE, Compton or energy cut)
                if ( particles.endsimu[ part_id ] == PARTICLE_DEAD &&  particles.E[ part_id ] != 0.0f )
                {
                    dose_record_standard( dosi, particles.E[ part_id ], pos.x,
                                          pos.y, pos.z );
                }

                // If electron particle has energy
                if ( electron.E != 0.0f )
                {
                    dose_record_standard( dosi, electron.E, pos.x,
                                          pos.y, pos.z );
                }
            } // geom boundary

        } // Analog

    } // TLE

    // store the new position
    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;
}


// Se TLE function
__host__ __device__ void VPIORTN::track_seTLE( ParticlesData &particles, VoxVolumeData<ui16> vol, COOHistoryMap coo_hist_map,
                                               DoseData dose, Mu_MuEn_Table mu_table,
                                               ui32 nb_of_rays, f32 edep_th, ui32 id )
{
    // Read an interaction position
    ui16 vox_x = coo_hist_map.x[ id ];
    ui16 vox_y = coo_hist_map.y[ id ];
    ui16 vox_z = coo_hist_map.z[ id ];

    // Nb of interaction and total energy
    ui32 nb_int = coo_hist_map.interaction[ id ];
    f32 mean_energy = coo_hist_map.energy[ id ] / f32( nb_int );

    // Total nb of rays is given by the ponderation of the nb of interactions
    nb_of_rays *= nb_int;

    // Weight in
    f32 win_init = 1 / f32( nb_of_rays );

    // vars DDA
    ui32 n;
    f32 length;

    f32 flength;
    f32 lx, ly, lz;
    f32 fxinc, fyinc, fzinc, fx, fy, fz;
    ui32 ix, iy, iz;
    f32 diffx, diffy, diffz;

    ui32 step = vol.nb_vox_x * vol.nb_vox_y;
    ui32 ind;

    // Rnd ray
    f32 phi, theta;
    f32xyz ray_p, ray_q, ray_d;
    f32 aabb_dist;

    // seTLE
    ui16 mat_id;
    ui32 E_index;
    f32 mu, mu_en, path_length;
    f32 win, wout, edep;

    // Pre-compute the energy index to access to the mu and mu_en tables
    E_index = binary_search ( mean_energy, mu_table.E_bins, mu_table.nb_bins );

    // Loop over raycasting
    ui32 iray=0; while ( iray < nb_of_rays )
    {
        // Generate a ray
        ray_p.x = f32(vox_x) + 0.5f;  // Center of the voxel
        ray_p.y = f32(vox_y) + 0.5f;  // Center of the voxel
        ray_p.z = f32(vox_z) + 0.5f;  // Center of the voxel

        phi = prng_uniform( particles, id );
        theta = prng_uniform( particles, id );
        phi  *= gpu_twopi;
        theta = acosf ( 1.0f - 2.0f*theta );
        ray_d.x = cosf( phi ) * sinf( theta );
        ray_d.y = sinf( phi ) * sinf( theta );
        ray_d.z = cosf( theta );

        // Get the second voxel point for the ray
        aabb_dist = hit_ray_AABB(ray_p, ray_d, 0, vol.nb_vox_x, 0, vol.nb_vox_y, 0, vol.nb_vox_z);
        ray_q = fxyz_add ( ray_p, fxyz_scale ( ray_d, aabb_dist ) );

        // DDA params
        diffx = floorf( ray_q.x ) - vox_x;
        diffy = floorf( ray_q.y ) - vox_y;
        diffz = floorf( ray_q.z ) - vox_z;

        lx = fabsf( diffx );
        ly = fabsf( diffy );
        lz = fabsf( diffz );

        length = fmaxf( ly, fmaxf( lx, lz ) );
        flength = 1.0 / length;

        fxinc = diffx * flength;
        fyinc = diffy * flength;
        fzinc = diffz * flength;

        fx = ray_p.x;
        fy = ray_p.y;
        fz = ray_p.z;

        // Path length of the increment step in mm
        path_length = powf( (fxinc*vol.spacing_x)*(fxinc*vol.spacing_x) +
                            (fyinc*vol.spacing_y)*(fyinc*vol.spacing_y) +
                            (fzinc*vol.spacing_z)*(fzinc*vol.spacing_z), 0.5f );

        // Init the particle weigth
        win = win_init;

        // DDA loop
        n = 0; while ( n < length )
        {
            ix = (ui32)fx; iy = (ui32)fy; iz = (ui32)fz;

            // if inside the volume
            if (fx >= 0 && fy >= 0 && fz >= 0 &&
                ix < vol.nb_vox_x && iy < vol.nb_vox_y && iz < vol.nb_vox_z
                && n != 0 )
            {

                // get index and accumulate
                ind = iz*step + iy*vol.nb_vox_x + ix;

                // Read material
                mat_id = vol.values[ ind ];

                // Get mu and mu_en for the current E
                if ( E_index == 0 )
                {
                    mu = mu_table.mu[ mat_id*mu_table.nb_bins ];
                    mu_en = mu_table.mu_en[ mat_id*mu_table.nb_bins ];
                }
                else
                {
                    mu = linear_interpolation( mu_table.E_bins[E_index-1],  mu_table.mu[mat_id*mu_table.nb_bins + E_index-1],
                                               mu_table.E_bins[E_index],    mu_table.mu[mat_id*mu_table.nb_bins + E_index],
                                               mean_energy );

                    mu_en = linear_interpolation( mu_table.E_bins[E_index-1],  mu_table.mu_en[mat_id*mu_table.nb_bins + E_index-1],
                                                  mu_table.E_bins[E_index],    mu_table.mu_en[mat_id*mu_table.nb_bins + E_index],
                                                  mean_energy );
                }

                // Compute the weight out
                wout = win * expf( -mu * path_length / 10.0 ); // Factor from GATE?? - JB

                // Compute the energy to drop
                edep = mean_energy * mu_en/mu * ( win - wout );

                // Drop energy
                ggems_atomic_add_f64( dose.edep, ind, f64( edep ) );
                ggems_atomic_add_f64( dose.edep_squared, ind, f64( edep) * f64( edep ) );
                ggems_atomic_add( dose.number_of_hits, ind, ui32 ( 1 ) );

                // Update the weight
                win = wout;

                // Energy cut
                if (edep <= edep_th) break;

            }

            // step the line
            fx += fxinc;
            fy += fyinc;
            fz += fzinc;
            ++n;
        }

        ++iray;
    } // Rays

}


/// KERNELS /////////////////////////////////


// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPIORTN::kernel_device_track_to_in( ParticlesData particles, f32 xmin, f32 xmax,
                                                    f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance )
{  
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, tolerance, id);
}


// Host Kernel that move particles to the voxelized volume boundary
void VPIORTN::kernel_host_track_to_in( ParticlesData particles, f32 xmin, f32 xmax,
                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 part_id )
{       
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, tolerance, part_id);
}

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPIORTN::kernel_device_track_to_out( ParticlesData particles,
                                                     VoxVolumeData<ui16> vol,
                                                     MaterialsTable materials,
                                                     PhotonCrossSectionTable photon_CS_table,
                                                     GlobalSimulationParametersData parameters,
                                                     DoseData dosi,
                                                     Mu_MuEn_Table mu_table,
                                                     HistoryMap hist_map )
{   
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    

#ifdef SKIP_VOXEL
    particles.geometry_id[ id ] = 999999999; // Init with unknow material id
#endif

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {
        VPIORTN::track_to_out( particles, vol, materials, photon_CS_table, parameters, dosi, mu_table, hist_map, id );
    }

}

// Host kernel that track particles within the voxelized volume until boundary
void VPIORTN::kernel_host_track_to_out( ParticlesData particles,
                                       VoxVolumeData<ui16> vol,
                                       MaterialsTable materials,
                                       PhotonCrossSectionTable photon_CS_table,
                                       GlobalSimulationParametersData parameters,
                                       DoseData dosi,
                                       Mu_MuEn_Table mu_table,
                                       HistoryMap hist_map )
{

    ui32 id=0;
    while ( id < particles.size )
    {
        #ifdef SKIP_VOXEL
            particles.geometry_id[ id ] = 999999999; // Init with unknow material id
        #endif

        // Stepping loop - Get out of loop only if the particle was dead and it was a primary
        while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
        {
            VPIORTN::track_to_out( particles, vol, materials, photon_CS_table, parameters, dosi, mu_table, hist_map, id );
        }
        ++id;
    }
}


// Device kernel that perform seTLE
__global__ void VPIORTN::kernel_device_seTLE( ParticlesData particles, VoxVolumeData<ui16> vol,
                                              COOHistoryMap coo_hist_map,
                                              DoseData dosi,
                                              Mu_MuEn_Table mu_table , ui32 nb_of_rays , f32 edep_th )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= coo_hist_map.nb_data ) return;

    VPIORTN::track_seTLE( particles, vol, coo_hist_map, dosi, mu_table, nb_of_rays, edep_th, id );
}

// Host kernel that perform seTLE
void VPIORTN::kernel_host_seTLE( ParticlesData particles, VoxVolumeData<ui16> vol,
                                 COOHistoryMap coo_hist_map,
                                 DoseData dosi,
                                 Mu_MuEn_Table mu_table , ui32 nb_of_rays , f32 edep_th )
{
    ui32 id = 0;
    while ( id < coo_hist_map.nb_data )
    {
        VPIORTN::track_seTLE( particles, vol, coo_hist_map, dosi, mu_table, nb_of_rays, edep_th, id );
        ++id;
    }
}

////:: Privates

bool VoxPhanIORTNav::m_check_mandatory()
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

// Init mu and mu_en table
void VoxPhanIORTNav::m_init_mu_table()
{
    // Load mu data
    f32 *energies  = new f32[mu_nb_energies];
    f32 *mu        = new f32[mu_nb_energies];
    f32 *mu_en     = new f32[mu_nb_energies];
    ui32 *mu_index = new ui32[mu_nb_elements];

    ui32 index_table = 0;
    ui32 index_data = 0;

    for (ui32 i= 0; i < mu_nb_elements; i++)
    {
        ui32 nb_energies = mu_nb_energy_bin[ i ];
        mu_index[ i ] = index_table;

        for (ui32 j = 0; j < nb_energies; j++)
        {
            energies[ index_table ] = mu_data[ index_data++ ];
            mu[ index_table ]       = mu_data[ index_data++ ];
            mu_en[ index_table ]    = mu_data[ index_data++ ];
            index_table++;
        }
    }

    // Build mu and mu_en according material
    m_mu_table.nb_mat = m_materials.data_h.nb_materials;
    m_mu_table.E_max = m_params.data_h.cs_table_max_E;
    m_mu_table.E_min = m_params.data_h.cs_table_min_E;
    m_mu_table.nb_bins = m_params.data_h.cs_table_nbins;

    HANDLE_ERROR( cudaMallocManaged( &(m_mu_table.E_bins), m_mu_table.nb_bins*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_mu_table.mu), m_mu_table.nb_mat*m_mu_table.nb_bins*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_mu_table.mu_en), m_mu_table.nb_mat*m_mu_table.nb_bins*sizeof( f32 ) ) );

    // Fill energy table with log scale
    f32 slope = log(m_mu_table.E_max / m_mu_table.E_min);
    ui32 i = 0;
    while (i < m_mu_table.nb_bins) {
        m_mu_table.E_bins[ i ] = m_mu_table.E_min * exp( slope * ( (f32)i / ( (f32)m_mu_table.nb_bins-1 ) ) ) * MeV;
        ++i;
    }

    // For each material and energy bin compute mu and muen
    ui32 imat = 0;
    ui32 abs_index, E_index, mu_index_E;
    ui32 iZ, Z;
    f32 energy, mu_over_rho, mu_en_over_rho, frac;
    while (imat < m_mu_table.nb_mat) {

        // for each energy bin
        i=0; while (i < m_mu_table.nb_bins) {

            // absolute index to store data within the table
            abs_index = imat*m_mu_table.nb_bins + i;

            // Energy value
            energy = m_mu_table.E_bins[ i ];

            // For each element of the material
            mu_over_rho = 0.0f; mu_en_over_rho = 0.0f;
            iZ=0; while (iZ < m_materials.data_h.nb_elements[ imat ]) {

                // Get Z and mass fraction
                Z = m_materials.data_h.mixture[ m_materials.data_h.index[ imat ] + iZ ];
                frac = m_materials.data_h.mass_fraction[ m_materials.data_h.index[ imat ] + iZ ];

                // Get energy index
                mu_index_E = mu_index_energy[ Z ];
                E_index = binary_search ( energy, energies, mu_index_E+mu_nb_energy_bin[ Z ], mu_index_E );

                // Get mu an mu_en from interpolation
                if ( E_index == mu_index_E )
                {
                    mu_over_rho += mu[ E_index ];
                    mu_en_over_rho += mu_en[ E_index ];
                }
                else
                {
                    mu_over_rho += frac * linear_interpolation(energies[E_index-1],  mu[E_index-1],
                                                               energies[E_index],    mu[E_index],
                                                               energy);
                    mu_en_over_rho += frac * linear_interpolation(energies[E_index-1],  mu_en[E_index-1],
                                                                  energies[E_index],    mu_en[E_index],
                                                                  energy);
                }
                ++iZ;
            }

            // Store values
            m_mu_table.mu[ abs_index ] = mu_over_rho * m_materials.data_h.density[ imat ] / (g/cm3);
            m_mu_table.mu_en[ abs_index ] = mu_en_over_rho * m_materials.data_h.density[ imat ] / (g/cm3);

            ++i;



        } // E bin

        ++imat;


    } // Mat

}

// Compress history map to be process by the GPU (in a non-sparse way)
void VoxPhanIORTNav::m_compress_history_map()
{
    // 1. count the number of non-zero
    ui32 ct = 0; ui32 i = 0; ui32 val_int;
    while ( i < m_phantom.data_h.number_of_voxels )
    {
        val_int = m_hist_map.interaction[ i++ ];
        if ( val_int ) ++ct;
    }
    m_coo_hist_map.nb_data = ct;
    GGcout << "Coo History map has " << ct << " non-zeros" << GGendl;

    // 2. init memory
    HANDLE_ERROR( cudaMallocManaged( &(m_coo_hist_map.x), ct * sizeof( ui16 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_coo_hist_map.y), ct * sizeof( ui16 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_coo_hist_map.z), ct * sizeof( ui16 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_coo_hist_map.energy), ct * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_coo_hist_map.interaction), ct * sizeof( ui32 ) ) );

    // 3. compressed data
    ui16 x, y, z;
    z = i = ct = 0;

    while ( z < m_phantom.data_h.nb_vox_z )
    {
        y = 0;
        while ( y < m_phantom.data_h.nb_vox_y )
        {
            x = 0;
            while ( x < m_phantom.data_h.nb_vox_x )
            {
                val_int = m_hist_map.interaction[ i ];
                if ( val_int )
                {
                    m_coo_hist_map.x[ ct ] = x;
                    m_coo_hist_map.y[ ct ] = y;
                    m_coo_hist_map.z[ ct ] = z;
                    m_coo_hist_map.interaction[ ct ] = val_int;
                    m_coo_hist_map.energy[ ct ] = m_hist_map.energy[ i ];
                    ++ct;
                }
                ++i;
                ++x;
            }
            ++y;
        }
        ++z;
    }

}

// return memory usage
ui64 VoxPhanIORTNav::m_get_memory_usage()
{
    ui64 mem = 0;

    // First the voxelized phantom
    mem += ( m_phantom.data_h.number_of_voxels * sizeof( ui16 ) );
    // Then material data
    mem += ( ( 3 * m_materials.data_h.nb_elements_total + 23 * m_materials.data_h.nb_materials ) * sizeof( f32 ) );
    // Then cross sections (gamma)
    ui64 n = m_cross_sections.photon_CS.data_h.nb_bins;
    ui64 k = m_cross_sections.photon_CS.data_h.nb_mat;
    mem += ( ( n + 3*n*k + 3*101*n ) * sizeof( f32 ) );
    // Cross section (electron)
    mem += ( n*k*7*sizeof( f32 ) );
    // Finally the dose map
    n = m_dose_calculator.dose.tot_nb_dosels;
    mem += ( 2*n*sizeof( f64 ) + n*sizeof( ui32 ) );
    mem += ( 20 * sizeof( f32 ) );

    // If TLE
    if ( m_flag_TLE )
    {
        n = m_mu_table.nb_bins;
        mem += ( n*k*2 * sizeof( f32 ) ); // mu and mu_en
        mem += ( n*sizeof( f32 ) );       // energies
    }

    // If seTLE
    if ( m_flag_TLE == seTLE )
    {
        mem += ( m_phantom.data_h.number_of_voxels * ( sizeof( ui32 ) + sizeof( f32 ) ) );
    }

    return mem;
}

////:: Main functions

VoxPhanIORTNav::VoxPhanIORTNav ()
{
    // Default doxel size (if 0 = same size to the phantom)
    m_dosel_size_x = 0;
    m_dosel_size_y = 0;
    m_dosel_size_z = 0;

    m_xmin = 0.0; m_xmax = 0.0;
    m_ymin = 0.0; m_ymax = 0.0;
    m_zmin = 0.0; m_zmax = 0.0;

    m_flag_TLE = analog;

    m_materials_filename = "";

    // Mu table
    m_mu_table.nb_mat = 0;
    m_mu_table.nb_bins = 0;
    m_mu_table.E_max = 0;
    m_mu_table.E_min = 0;

    m_mu_table.E_bins = NULL;
    m_mu_table.mu = NULL;
    m_mu_table.mu_en = NULL;

    m_hist_map.interaction = NULL;
    m_hist_map.energy = NULL;

    m_coo_hist_map.x = NULL;
    m_coo_hist_map.y = NULL;
    m_coo_hist_map.z = NULL;
    m_coo_hist_map.energy = NULL;
    m_coo_hist_map.interaction = NULL;
    m_coo_hist_map.nb_data = 0;

    m_mu_table.flag = analog; // Not used

    set_name( "VoxPhanIORTNav" );
}

void VoxPhanIORTNav::track_to_in( Particles particles )
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

        VPIORTN::kernel_device_track_to_in<<<grid, threads>>> ( particles.data_d, m_phantom.data_d.xmin, m_phantom.data_d.xmax,
                                                                               m_phantom.data_d.ymin, m_phantom.data_d.ymax,
                                                                               m_phantom.data_d.zmin, m_phantom.data_d.zmax,
                                                                               m_params.data_d.geom_tolerance );
        cuda_error_check ( "Error ", " Kernel_VoxPhanIORT (track to in)" );
        cudaDeviceSynchronize();
    }

}

void VoxPhanIORTNav::track_to_out ( Particles particles )
{
    //
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        VPIORTN::kernel_host_track_to_out( particles.data_h, m_phantom.data_h,
                                           m_materials.data_h, m_cross_sections.photon_CS.data_h,
                                           m_params.data_h, m_dose_calculator.dose,
                                           m_mu_table, m_hist_map );

        // Apply seTLE: splitting and determinstic raycasting
        if( m_flag_TLE == seTLE )
        {
            f64 t_start = get_time();
            m_compress_history_map();
            GGcout_time ( "Compress history map", get_time()-t_start );

            t_start = get_time();
            VPIORTN::kernel_host_seTLE( particles.data_h, m_phantom.data_h,
                                        m_coo_hist_map, m_dose_calculator.dose,
                                        m_mu_table, 100, 0.0 *eV );
            GGcout_time ( "Raycast", get_time()-t_start );
            GGnewline();

        }

    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {       
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;
        VPIORTN::kernel_device_track_to_out<<<grid, threads>>> ( particles.data_d, m_phantom.data_d, m_materials.data_d,
                                                              m_cross_sections.photon_CS.data_d,
                                                              m_params.data_d, m_dose_calculator.dose,
                                                              m_mu_table, m_hist_map );
        cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to out)" );             
        cudaDeviceSynchronize();

        // Apply seTLE: splitting and determinstic raycasting
        if( m_flag_TLE == seTLE )
        {
            f64 t_start = get_time();
            m_compress_history_map();
            GGcout_time ( "Compress history map", get_time()-t_start );

            threads.x = m_params.data_h.gpu_block_size;//
            grid.x = ( m_coo_hist_map.nb_data + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

            t_start = get_time();
            VPIORTN::kernel_device_seTLE<<<grid, threads>>> ( particles.data_d, m_phantom.data_d,
                                                              m_coo_hist_map, m_dose_calculator.dose,
                                                              m_mu_table, 1000, 0.0 *eV );
            cuda_error_check ( "Error ", " Kernel_device_seTLE" );
            cudaDeviceSynchronize();
            GGcout_time ( "Raycast", get_time()-t_start );
            GGnewline();
        }
    }
        
}

void VoxPhanIORTNav::load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
{
    m_phantom.load_from_mhd ( filename, range_mat_name );
}

void VoxPhanIORTNav::write ( std::string filename )
{
//     m_dose_calculator.m_copy_dose_gpu2cpu();

    m_dose_calculator.write ( filename );
}

// Export density values of the phantom
void VoxPhanIORTNav::export_density_map( std::string filename )
{
    ui32 N = m_phantom.data_h.number_of_voxels;
    f32 *density = new f32[ N ];
    ui32 i = 0;
    while (i < N)
    {
        density[ i ] = m_materials.data_h.density[ m_phantom.data_h.values[ i ] ];
        ++i;
    }

    f32xyz offset = make_f32xyz( m_phantom.data_h.off_x, m_phantom.data_h.off_y, m_phantom.data_h.off_z );
    f32xyz voxsize = make_f32xyz( m_phantom.data_h.spacing_x, m_phantom.data_h.spacing_y, m_phantom.data_h.spacing_z );
    ui32xyz nbvox = make_ui32xyz( m_phantom.data_h.nb_vox_x, m_phantom.data_h.nb_vox_y, m_phantom.data_h.nb_vox_z );

    ImageIO *im_io = new ImageIO;
    im_io->write_3D( filename, density, nbvox, offset, voxsize );
    delete im_io;
}

// Export materials index of the phantom
void VoxPhanIORTNav::export_materials_map( std::string filename )
{
    f32xyz offset = make_f32xyz( m_phantom.data_h.off_x, m_phantom.data_h.off_y, m_phantom.data_h.off_z );
    f32xyz voxsize = make_f32xyz( m_phantom.data_h.spacing_x, m_phantom.data_h.spacing_y, m_phantom.data_h.spacing_z );
    ui32xyz nbvox = make_ui32xyz( m_phantom.data_h.nb_vox_x, m_phantom.data_h.nb_vox_y, m_phantom.data_h.nb_vox_z );

    ImageIO *im_io = new ImageIO;
    im_io->write_3D( filename, m_phantom.data_h.values, nbvox, offset, voxsize );
    delete im_io;
}

// Export history map from seTLE
void VoxPhanIORTNav::export_history_map( std::string filename )
{
    if ( m_flag_TLE == seTLE )
    {
        f32xyz offset = make_f32xyz( m_phantom.data_h.off_x, m_phantom.data_h.off_y, m_phantom.data_h.off_z );
        f32xyz voxsize = make_f32xyz( m_phantom.data_h.spacing_x, m_phantom.data_h.spacing_y, m_phantom.data_h.spacing_z );
        ui32xyz nbvox = make_ui32xyz( m_phantom.data_h.nb_vox_x, m_phantom.data_h.nb_vox_y, m_phantom.data_h.nb_vox_z );


        // Create an IO object
        ImageIO *im_io = new ImageIO;

        std::string format = im_io->get_extension( filename );
        filename = im_io->get_filename_without_extension( filename );

        // Convert Edep from f64 to f32
        ui32 tot = m_dose_calculator.dose.nb_dosels.x * m_dose_calculator.dose.nb_dosels.y * m_dose_calculator.dose.nb_dosels.z;
        f32 *f32edep = new f32[ tot ];
        ui32 i=0; while ( i < tot )
        {
            f32edep[ i ] = (f32)m_dose_calculator.dose.edep[ i ];
            ++i;
        }

        // Get output name
        std::string int_out( filename + "-Interaction." + format );
        std::string energy_out( filename + "-Energies." + format );

        // Export
        im_io->write_3D( int_out, m_hist_map.interaction, nbvox, offset, voxsize );
        im_io->write_3D( energy_out, m_hist_map.energy, nbvox, offset, voxsize );
    }
    else
    {
        GGwarn << "History map export is only available while using seTLE option!" << GGendl;
    }
}

void VoxPhanIORTNav::initialize ( GlobalSimulationParameters params )
{
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanIORT: missing parameters." );
        exit_simulation();
    }

    // Params
    m_params = params;

    // Phantom
    m_phantom.set_name( "VoxPhanIORTNav" );
    m_phantom.initialize( params );

    // Materials table
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize( m_phantom.list_of_materials, params );    

    // Cross Sections
    m_cross_sections.initialize( m_materials, params );

    // Init dose map
    m_dose_calculator.set_voxelized_phantom( m_phantom );
    m_dose_calculator.set_materials( m_materials );
    m_dose_calculator.set_dosel_size( m_dosel_size_x, m_dosel_size_y, m_dosel_size_z );
    m_dose_calculator.set_voi( m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax );
    m_dose_calculator.initialize( m_params ); // CPU&GPU

    // If TLE init mu and mu_en table
    if ( m_flag_TLE )
    {
        m_init_mu_table();
    }

    // if seTLE init history map
    if ( m_flag_TLE == seTLE )
    {
        HANDLE_ERROR( cudaMallocManaged( &(m_hist_map.interaction), m_phantom.data_h.number_of_voxels * sizeof( ui32 ) ) );
        HANDLE_ERROR( cudaMallocManaged( &(m_hist_map.energy), m_phantom.data_h.number_of_voxels * sizeof( f32 ) ) );

        ui32 i=0; while (i < m_phantom.data_h.number_of_voxels )
        {
            m_hist_map.interaction[ i ] = 0;
            m_hist_map.energy[ i ] = 0.0;
            ++i;
        }
    }

    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanIORTNav", mem);
    }

}

void VoxPhanIORTNav::calculate_dose_to_water()
{
    m_dose_calculator.calculate_dose_to_water();

}

void VoxPhanIORTNav::calculate_dose_to_phantom()
{
    m_dose_calculator.calculate_dose_to_phantom();

}

void VoxPhanIORTNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}

/*
void VoxPhanIORTNav::set_doxel_size( f32 sizex, f32 sizey, f32 sizez )
{
    m_doxel_size_x = sizex;
    m_doxel_size_y = sizey;
    m_doxel_size_z = sizez;
}
*/

/*
void VoxPhanIORTNav::set_volume_of_interest( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{
    m_xmin = xmin; m_xmax = xmax;
    m_ymin = ymin; m_ymax = ymax;
    m_zmin = zmin; m_zmax = zmax;
}
*/

void VoxPhanIORTNav::set_kerma_estimator( std::string kind )
{
    // Transform the name of the process in small letter
    std::transform( kind.begin(), kind.end(), kind.begin(), ::tolower );

    if ( kind == "tle" )
    {
        m_flag_TLE = TLE;
        m_mu_table.flag = TLE; // Use TLE
    }
    else if ( kind == "setle" )
    {
        m_flag_TLE = seTLE;
        m_mu_table.flag = seTLE; // Use seTLE
    }
    else if ( kind == "analog" )
    {
        m_flag_TLE = analog;
        m_mu_table.flag = analog; // Use analog
    }
    else
    {
        GGcerr << "Track length estimator not recognized: '" << kind << "'!" << GGendl;
        exit_simulation();
    }
}

VoxVolumeData<f32> * VoxPhanIORTNav::get_dose_map()
{
    return m_dose_calculator.get_dose_map();
}

AabbData VoxPhanIORTNav::get_bounding_box()
{
    AabbData box;

    box.xmin = m_phantom.data_h.xmin;
    box.xmax = m_phantom.data_h.xmax;
    box.ymin = m_phantom.data_h.ymin;
    box.ymax = m_phantom.data_h.ymax;
    box.zmin = m_phantom.data_h.zmin;
    box.zmax = m_phantom.data_h.zmax;

    return box;
}


#undef DEBUG

#endif
