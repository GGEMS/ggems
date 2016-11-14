// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_gtrack_nav.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 08/11/2016
 *
 *
 *
 */

#ifndef VOX_PHAN_GTRACK_NAV_CU
#define VOX_PHAN_GTRACK_NAV_CU

#include "vox_phan_gtrack_nav.cuh"

////// HOST-DEVICE GPU Codes ////////////////////////////////////////////
/*
__host__ __device__ void VPGTN::_track_to_out( ParticlesData particles,
                                              VoxVolumeData<ui16> vol,
                                              GTrackModelData model,

                                              GlobalSimulationParametersData parameters,

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

    //// Get step distance ///////////////////////////////////////

    // Search the energy index to read CS
    f32 energy = particles.E[ part_id ];
    ui32 E_index = binary_search( energy, model.bin_energy, model.nb_energy_bins );

    // Get index in table
    ui32 read_index = E_index * model.nb_bins;

    if (part_id==12853)
    {
        printf("id %i: E %f  Eindex %i  valE %f  readIndex %i\n", part_id, energy, E_index, model.bin_energy[E_index],
               read_index );
    }

    // Fetch step value
    f32 rndm = prng_uniform( particles, part_id );
    ui32 bin_pos = binary_search_left_offset( rndm, model.cdf_step, model.nb_bins, read_index );

#ifdef DEBUG
    assert( bin_pos < model.nb_bins );
#endif

    f32 next_interaction_distance = model.bin_step[ bin_pos ];
    f32 dist = next_interaction_distance;
    ui32 bin_dist = bin_pos;
    f32 rnd_pos = rndm;

//    printf("id %i: rndm %f binPos %i next step %f\n", part_id, rndm, bin_pos, next_interaction_distance);

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

    ui8 exit = false;
    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters.geom_tolerance; // Overshoot
        exit = true;
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
//        printf("id %i: E %f OutOfWorld\n", part_id, energy);
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    // If boundary
    if ( exit )
    {
        if (part_id==12853)
        {
            printf("id %i: E %f Boundary dist %f pos %f %f %f (posBin %i val %f)\n", part_id, energy, boundary_distance,
                   pos.x, pos.y, pos.z, bin_pos, dist);
        }
        return;
    }

    // Scattering
    rndm = prng_uniform( particles, part_id );
    bin_pos = binary_search_left_offset( rndm, model.cdf_scatter, model.nb_bins, read_index );

#ifdef DEBUG
    assert( bin_pos < model.nb_bins );
#endif

    f32 theta = model.bin_scatter[ bin_pos ];
    f32 phi = prng_uniform( particles, part_id ) * gpu_twopi;

    // Get scattered gamma
    f32xyz gamDir1 = make_f32xyz( sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta) );
    gamDir1 = rotateUz(gamDir1, make_f32xyz( particles.dx[part_id], particles.dx[part_id], particles.dx[part_id] ) );
    gamDir1 = fxyz_unit( gamDir1 );

    // Get new energy
    rndm = prng_uniform( particles, part_id );
    bin_pos = binary_search_left_offset( rndm, model.cdf_edep, model.nb_bins, read_index );

#ifdef DEBUG
    assert( bin_pos < model.nb_bins );
#endif

    energy = energy - model.bin_edep[ bin_pos ];

    // Update gamma
    particles.dx[ part_id ] = gamDir1.x;
    particles.dy[ part_id ] = gamDir1.y;
    particles.dz[ part_id ] = gamDir1.z;
    particles.E[ part_id ] = energy;

    // Energy cut
    if ( energy <= 1.0 *keV )
    {
        particles.endsimu[ part_id ] = PARTICLE_DEAD;
    }

    if (part_id==12853)
    {
        printf( "id %i: next step %f  angle %f  newE %f (rnd %f posBin %i valBin %f dist %f)\n", part_id, next_interaction_distance,
                theta, energy, rnd_pos, bin_dist, model.cdf_step[ read_index + bin_dist ] , dist );
    }

}
*/


__host__ __device__ void VPGTN::track_to_out_uncorrelated_model( ParticlesData particles,
                                                                VoxVolumeData<ui16> vol,
                                                                GTrackUncorrelatedModelData model,
                                                                /*MaterialsTable materials,*/
                                                                /*PhotonCrossSectionTable photon_CS_table,*/
                                                                GlobalSimulationParametersData parameters,
                                                                /*DoseData dosi,*/
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
    //ui16 mat_id = vol.values[ index_phantom.w ];

    //// Get step distance ///////////////////////////////////////

    // Search the energy index to read CS
    f32 energy = particles.E[ part_id ];
    ui32 E_index = ui32( (energy - model.min_E) / model.di_energy );
//    printf("id %i: energy %f diE %f binPos %i ValE %f\n", part_id, energy, model.di_energy, E_index, model.bin_energy[E_index]);



    // Get index in table
    ui32 read_index = E_index * model.nb_lut_bins;

    // Fetch step value
    f32 rndm = prng_uniform( particles, part_id );
    ui32 bin_pos = ui32( rndm / model.di_lut );
//    printf("id %i: rndm %f binPos %i\n", part_id, rndm, bin_pos);

    bin_pos = model.lcdf_step[ read_index + bin_pos ];

//    printf("id %i: rndm %f GblPos %i StepPos %i\n", part_id, rndm, read_index+bin_pos, bin_pos);

#ifdef DEBUG
    assert( bin_pos < model.nb_bins );
#endif

    f32 next_interaction_distance = model.bin_step[ bin_pos ];
//    f32 dist = next_interaction_distance;
//    ui32 bin_dist = bin_pos;
//    f32 rnd_pos = rndm;

//    printf("id %i: rndm %f binPos %i next step %f\n", part_id, rndm, bin_pos, next_interaction_distance);

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

    ui8 exit = false;
    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters.geom_tolerance; // Overshoot
        exit = true;
    }

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, boundary_distance + parameters.geom_tolerance ) );

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
//        printf("id %i: E %f OutOfWorld\n", part_id, energy);
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    // If boundary
    if ( exit )
    {
//        if (part_id==12853)
//        {
//            printf("id %i: E %f Boundary dist %f pos %f %f %f (posBin %i val %f)\n", part_id, energy, boundary_distance,
//                   pos.x, pos.y, pos.z, bin_pos, dist);
//        }

//        printf("id %i: E %f Boundary dist %f pos %f %f %f (posBin %i val %f)\n", part_id, energy, boundary_distance,
//                          pos.x, pos.y, pos.z, bin_pos, dist);


        return;
    }

    // Scattering
    rndm = prng_uniform( particles, part_id );
    //bin_pos = binary_search_left_offset( rndm, model.cdf_scatter, model.nb_bins, read_index );
    bin_pos = ui32( rndm / model.di_lut );
    bin_pos = model.lcdf_scatter[ read_index + bin_pos ];

#ifdef DEBUG
    assert( bin_pos < model.nb_bins );
#endif

    f32 theta = model.bin_scatter[ bin_pos ];
    f32 phi = prng_uniform( particles, part_id ) * gpu_twopi;

    // Get scattered gamma
    f32xyz gamDir1 = make_f32xyz( sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta) );
    gamDir1 = rotateUz(gamDir1, make_f32xyz( particles.dx[part_id], particles.dx[part_id], particles.dx[part_id] ) );
    gamDir1 = fxyz_unit( gamDir1 );

    // Get new energy
    rndm = prng_uniform( particles, part_id );
    //bin_pos = binary_search_left_offset( rndm, model.cdf_edep, model.nb_bins, read_index );
    bin_pos = ui32( rndm / model.di_lut );
    bin_pos = model.lcdf_edep[ read_index + bin_pos ];

#ifdef DEBUG
    assert( bin_pos < model.nb_bins );
#endif

    energy = energy - model.bin_edep[ bin_pos ];

    // Update gamma
    particles.dx[ part_id ] = gamDir1.x;
    particles.dy[ part_id ] = gamDir1.y;
    particles.dz[ part_id ] = gamDir1.z;
    particles.E[ part_id ] = energy;

    // Energy cut
    if ( energy <= 1.0 *keV )
    {
        particles.endsimu[ part_id ] = PARTICLE_DEAD;
    }

//    if (part_id==12853)
//    {
//        printf( "id %i: next step %f  angle %f  newE %f (rnd %f posBin %i valBin %f dist %f)\n", part_id, next_interaction_distance,
//                theta, energy, rnd_pos, bin_dist, model.cdf_step[ read_index + bin_dist ] , dist );
//    }

//    printf( "id %i: next step %f  angle %f  newE %f (rnd %f posBin %i valBin %f dist %f)\n", part_id, next_interaction_distance,
//            theta, energy, rnd_pos, bin_dist, model.lcdf_step[ read_index + bin_dist ], dist );

}





/// KERNELS /////////////////////////////////


// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPGTN::kernel_device_track_to_in( ParticlesData particles, f32 xmin, f32 xmax,
                                                  f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance )
{  
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, tolerance, id);
}

/*
// Host Kernel that move particles to the voxelized volume boundary
void VPGTN::kernel_host_track_to_in( ParticlesData particles, f32 xmin, f32 xmax,
                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 part_id )
{       
    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, tolerance, part_id);
}
*/

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPGTN::kernel_device_track_to_out_uncorrelated_model( ParticlesData particles,
                                                   VoxVolumeData<ui16> vol,
                                                   /* MaterialsTable materials, */
                                                   /* PhotonCrossSectionTable photon_CS_table, */
                                                   GTrackUncorrelatedModelData model,
                                                   GlobalSimulationParametersData parameters
                                                   /*DoseData dosi*/ )
{   
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    

    ui32 ct=0;

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {
        VPGTN::track_to_out_uncorrelated_model( particles, vol, model, parameters, id );
        ct++;

        if (ct > 1000)
        {
            printf("Id %i Inf loop E=%f\n", id, particles.E[id]);
            break;
        }
    }

//    printf("%i steps\n", ct);

}


/*
// Host kernel that track particles within the voxelized volume until boundary
void VPGTN::kernel_host_track_to_out( ParticlesData particles,
                                      VoxVolumeData<ui16> vol,

                                      GTrackModelData model,
                                      GlobalSimulationParametersData parameters
                                       )
{

    ui32 id=0;
    while ( id < particles.size )
    {
        // Stepping loop - Get out of loop only if the particle was dead and it was a primary
        while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
        {
            VPGTN::track_to_out( particles, vol, model, parameters, id );
        }
        ++id;
    }
}
*/
////:: Privates

bool VoxPhanGTrackNav::m_check_mandatory()
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
ui64 VoxPhanGTrackNav::m_get_memory_usage()
{
    ui64 mem = 0;
/*
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
*/
    return mem;
}

////:: Main functions

VoxPhanGTrackNav::VoxPhanGTrackNav ()
{
    // Default doxel size (if 0 = same size to the phantom)
    m_dosel_size_x = 0;
    m_dosel_size_y = 0;
    m_dosel_size_z = 0;

    m_xmin = 0.0; m_xmax = 0.0;
    m_ymin = 0.0; m_ymax = 0.0;
    m_zmin = 0.0; m_zmax = 0.0;

    m_materials_filename = "";

    set_name( "VoxPhanGTrackNav" );
}

void VoxPhanGTrackNav::track_to_in( Particles particles )
{

    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
//        ui32 id=0;
//        while ( id<particles.size )
//        {
//            VPDN::kernel_host_track_to_in ( particles.data_h, m_phantom.data_h.xmin, m_phantom.data_h.xmax,
//                                            m_phantom.data_h.ymin, m_phantom.data_h.ymax,
//                                            m_phantom.data_h.zmin, m_phantom.data_h.zmax,
//                                            m_params.data_h.geom_tolerance,
//                                            id );
//            ++id;
//        }
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {

        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

        VPGTN::kernel_device_track_to_in<<<grid, threads>>> ( particles.data_d, m_phantom.data_d.xmin, m_phantom.data_d.xmax,
                                                                               m_phantom.data_d.ymin, m_phantom.data_d.ymax,
                                                                               m_phantom.data_d.zmin, m_phantom.data_d.zmax,
                                                                               m_params.data_d.geom_tolerance );
        cuda_error_check ( "Error ", " Kernel_VoxPhanGTrackNav (track to in)" );
        cudaDeviceSynchronize();
    }

}

void VoxPhanGTrackNav::track_to_out ( Particles particles )
{
    //
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
//        VPGTN::kernel_host_track_to_out( particles.data_h, m_phantom.data_h,
//                                         m_gtrack_model,
//                                         m_params.data_h );
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {

        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;
        VPGTN::kernel_device_track_to_out_uncorrelated_model<<<grid, threads>>> ( particles.data_d, m_phantom.data_d,
                                                                                  m_gtrack_uncorrelated_model,
                                                                                  m_params.data_d );

        cuda_error_check ( "Error ", " Kernel_VoxPhanGTrackNav (track to out)" );
        cudaDeviceSynchronize();        
    }
        
}

void VoxPhanGTrackNav::load_phantom_from_mhd( std::string filename, std::string range_mat_name )
{
    m_phantom.load_from_mhd( filename, range_mat_name );
}

void VoxPhanGTrackNav::load_gtrack_uncorrelated_model( std::string filename )
{
    // TODO, open MHD header to read the params
    ui32 nb_bins = 100;
    ui32 nb_energy_bins = 100;
    ui32 nb_lut = 10000;

    m_gtrack_uncorrelated_model.nb_bins = nb_bins;
    m_gtrack_uncorrelated_model.nb_energy_bins = nb_energy_bins;
    m_gtrack_uncorrelated_model.nb_lut_bins = nb_lut;

    m_gtrack_uncorrelated_model.di_lut = 1.0f / f32(nb_lut - 1);
    m_gtrack_uncorrelated_model.di_energy = ( 50 *keV - 10 *keV ) / f32( nb_energy_bins - 1);
    m_gtrack_uncorrelated_model.min_E = 10 *keV;

    // Allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_uncorrelated_model.bin_energy), nb_energy_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_uncorrelated_model.bin_step), nb_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_uncorrelated_model.bin_edep), nb_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_uncorrelated_model.bin_scatter), nb_bins * sizeof( f32 ) ) );
/*
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_model.cdf_step), nb_bins*nb_energy_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_model.cdf_edep), nb_bins*nb_energy_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_model.cdf_scatter), nb_bins*nb_energy_bins * sizeof( f32 ) ) );
*/
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_uncorrelated_model.lcdf_step), nb_lut*nb_energy_bins * sizeof( ui16 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_uncorrelated_model.lcdf_edep), nb_lut*nb_energy_bins * sizeof( ui16 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_gtrack_uncorrelated_model.lcdf_scatter), nb_lut*nb_energy_bins * sizeof( ui16 ) ) );

    // Open data
    FILE *pfile = fopen(filename.c_str(), "rb");

    if ( !pfile ) {
        GGcerr << "Error when loading mhd file: " << filename << GGendl;
        exit_simulation();
    }

    // Load data
    fread( m_gtrack_uncorrelated_model.bin_energy, sizeof( f32 ), nb_energy_bins, pfile );
    fread( m_gtrack_uncorrelated_model.bin_step, sizeof( f32 ), nb_bins, pfile );
    fread( m_gtrack_uncorrelated_model.bin_edep, sizeof( f32 ), nb_bins, pfile );
    fread( m_gtrack_uncorrelated_model.bin_scatter, sizeof( f32 ), nb_bins, pfile );
/*
    fread( m_gtrack_model.cdf_step, sizeof( f32 ), nb_bins*nb_energy_bins, pfile );
    fread( m_gtrack_model.cdf_edep, sizeof( f32 ), nb_bins*nb_energy_bins, pfile );
    fread( m_gtrack_model.cdf_scatter, sizeof( f32 ), nb_bins*nb_energy_bins, pfile );
*/

    fread( m_gtrack_uncorrelated_model.lcdf_step, sizeof( ui16 ), nb_lut*nb_energy_bins, pfile );
    fread( m_gtrack_uncorrelated_model.lcdf_edep, sizeof( ui16 ), nb_lut*nb_energy_bins, pfile );
    fread( m_gtrack_uncorrelated_model.lcdf_scatter, sizeof( ui16 ), nb_lut*nb_energy_bins, pfile );

    fclose( pfile );

}


//void VoxPhanGTrackNav::write ( std::string filename )
//{
//    m_dose_calculator.write ( filename );
//}

void VoxPhanGTrackNav::initialize ( GlobalSimulationParameters params )
{
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanGTrackNav: missing parameters." );
        exit_simulation();
    }

    // Params
    m_params = params;

    // Phantom
    m_phantom.set_name( "VoxPhanGTrackNav" );
    m_phantom.initialize( params );

//    // Materials table
//    m_materials.load_materials_database( m_materials_filename );
//    m_materials.initialize( m_phantom.list_of_materials, params );

//    // Cross Sections
//    m_cross_sections.initialize( m_materials, params );

//    // Init dose map
//    m_dose_calculator.set_voxelized_phantom( m_phantom );
//    m_dose_calculator.set_materials( m_materials );
//    m_dose_calculator.set_dosel_size( m_dosel_size_x, m_dosel_size_y, m_dosel_size_z );
//    m_dose_calculator.set_voi( m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax );
//    m_dose_calculator.initialize( m_params ); // CPU&GPU

    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanGTrackNav", mem);
    }

}

//void VoxPhanGTrackNav::calculate_dose_to_water()
//{
//    m_dose_calculator.calculate_dose_to_water();

//}

//void VoxPhanGTrackNav::calculate_dose_to_phantom()
//{
//    m_dose_calculator.calculate_dose_to_phantom();

//}

void VoxPhanGTrackNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}

//VoxVolumeData<f32> * VoxPhanIORTNav::get_dose_map()
//{
//    return m_dose_calculator.get_dose_map();
//}

AabbData VoxPhanGTrackNav::get_bounding_box()
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
