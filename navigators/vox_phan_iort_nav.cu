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

////:: GPU Codes

__host__ __device__ void VPIORTN::track_to_out( ParticlesData &particles,
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

        /// Drop energy ////////////

        // If gamma particle is dead (PE, Compton or energy cut)
        if ( particles.endsimu[ part_id ] == PARTICLE_DEAD &&  particles.E[ part_id ] != 0.0f )
        {
            dose_record_standard( dosi, particles.E[ part_id ], particles.px[ part_id ],
                                  particles.py[ part_id ], particles.pz[ part_id ] );
        }

        // If electron particle has energy
        if ( electron.E != 0.0f )
        {
            dose_record_standard( dosi, electron.E, particles.px[ part_id ],
                                  particles.py[ part_id ], particles.pz[ part_id ] );
        }

    } // discrete process

}

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
                                                   VoxVolumeData vol,
                                                   MaterialsTable materials,
                                                   PhotonCrossSectionTable photon_CS_table,
                                                   GlobalSimulationParametersData parameters,
                                                   DoseData dosi )
{   
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;    

    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {
        VPIORTN::track_to_out( particles, vol, materials, photon_CS_table, parameters, dosi, id );
    }

}

// Host kernel that track particles within the voxelized volume until boundary
void VPIORTN::kernel_host_track_to_out( ParticlesData particles,
                                      VoxVolumeData vol,
                                      MaterialsTable materials,
                                      PhotonCrossSectionTable photon_CS_table,
                                      GlobalSimulationParametersData parameters,
                                      DoseData dosi,
                                      ui32 id )
{
    // Stepping loop - Get out of loop only if the particle was dead and it was a primary
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {
        VPIORTN::track_to_out( particles, vol, materials, photon_CS_table, parameters, dosi, id );
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

// return memory usage
ui64 VoxPhanIORTNav::m_get_memory_usage()
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

VoxPhanIORTNav::VoxPhanIORTNav ()
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

void VoxPhanIORTNav::track_to_in ( Particles particles )
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
        cudaThreadSynchronize();
    }

}

void VoxPhanIORTNav::track_to_out ( Particles particles )
{
    //
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {

        ui32 id=0;
        while ( id<particles.size )
        {
            VPIORTN::kernel_host_track_to_out( particles.data_h, m_phantom.data_h,
                                            m_materials.data_h, m_cross_sections.photon_CS.data_h,
                                            m_params.data_h, m_dose_calculator.dose.data_h,
                                            id );
            ++id;
        }
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {       
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;//
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;
        cudaThreadSynchronize();
        VPIORTN::kernel_device_track_to_out<<<grid, threads>>> ( particles.data_d, m_phantom.data_d, m_materials.data_d,
                                                              m_cross_sections.photon_CS.data_d,
                                                              m_params.data_d, m_dose_calculator.dose.data_d );
        cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to out)" );
        
        cudaThreadSynchronize();
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

    ImageReader::record3Dimage (filename, density, offset, voxsize, nbvox );
}

// Export materials index of the phantom
void VoxPhanIORTNav::export_materials_map( std::string filename )
{
    f32xyz offset = make_f32xyz( m_phantom.data_h.off_x, m_phantom.data_h.off_y, m_phantom.data_h.off_z );
    f32xyz voxsize = make_f32xyz( m_phantom.data_h.spacing_x, m_phantom.data_h.spacing_y, m_phantom.data_h.spacing_z );
    ui32xyz nbvox = make_ui32xyz( m_phantom.data_h.nb_vox_x, m_phantom.data_h.nb_vox_y, m_phantom.data_h.nb_vox_z );

    ImageReader::record3Dimage (filename, m_phantom.data_h.values, offset, voxsize, nbvox );
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
    m_dose_calculator.set_doxel_size( m_doxel_size_x, m_doxel_size_y, m_doxel_size_z );
    m_dose_calculator.set_voi( m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax );
    m_dose_calculator.initialize( m_params ); // CPU&GPU

    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanIORTNav", mem);
    }

}

void VoxPhanIORTNav::calculate_dose_to_water(){

    m_dose_calculator.calculate_dose_to_water();

}

void VoxPhanIORTNav::calculate_dose_to_phantom(){

    m_dose_calculator.calculate_dose_to_phantom();

}

void VoxPhanIORTNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}

void VoxPhanIORTNav::set_doxel_size( f32 sizex, f32 sizey, f32 sizez )
{
    m_doxel_size_x = sizex;
    m_doxel_size_y = sizey;
    m_doxel_size_z = sizez;
}

void VoxPhanIORTNav::set_volume_of_interest( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{
    m_xmin = xmin; m_xmax = xmax;
    m_ymin = ymin; m_ymax = ymax;
    m_zmin = zmin; m_zmax = zmax;
}

#undef DEBUG

#endif
