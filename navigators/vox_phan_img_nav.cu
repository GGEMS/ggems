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

#ifndef VOX_PHAN_IMG_NAV_CU
#define VOX_PHAN_IMG_NAV_CU

#include "vox_phan_img_nav.cuh"

////:: GPU Codes

__device__ void VPIN::track_to_out( ParticlesData particles,
                                    const VoxVolumeData<ui16> *vol,
                                    const MaterialsData *materials,
                                    const PhotonCrossSectionData *photon_CS_table,
                                    const GlobalSimulationParametersData *parameters,
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
    f32 next_interaction_distance = particles.next_interaction_distance[part_id];
    ui8 next_discrete_process = particles.next_discrete_process[part_id];

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
    particles.tof[part_id] += c_light * next_interaction_distance;

    // store new position
    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance( pos, vol->xmin, vol->xmax, vol->ymin, vol->ymax,
                                          vol->zmin, vol->zmax, parameters->geom_tolerance ) )
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

        // If the process is PHOTON_COMPTON or PHOTON_RAYLEIGH the scatter
        // order is incremented
        if( next_discrete_process == PHOTON_COMPTON
                || next_discrete_process == PHOTON_RAYLEIGH )
        {
            particles.scatter_order[ part_id ] += 1;
        }

        //// Here e- are not tracked, and lost energy not drop
        //// Energy cut
        if ( particles.E[ part_id ] <= materials->photon_energy_cut[ mat_id ])
        {
            // kill without mercy (energy not drop)
            particles.endsimu[part_id] = PARTICLE_DEAD;
            return;
        }
    }
}

// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPIN::kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                                  f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 geom_tolerance )
{

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    transport_track_to_in_AABB( particles, xmin, xmax, ymin, ymax, zmin, zmax, geom_tolerance, id );

}

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPIN::kernel_device_track_to_out ( ParticlesData particles,
                                                   const VoxVolumeData<ui16> *vol,
                                                   const MaterialsData *materials,
                                                   const PhotonCrossSectionData *photon_CS_table,
                                                   const GlobalSimulationParametersData *parameters )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {        
        VPIN::track_to_out( particles, vol, materials, photon_CS_table, parameters, id );
    }

}

////:: Privates

bool VoxPhanImgNav::m_check_mandatory()
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
ui64 VoxPhanImgNav::m_get_memory_usage()
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

    return mem;
}

////:: Main functions

VoxPhanImgNav::VoxPhanImgNav()
{
    m_materials_filename = "";
    set_name( "VoxPhanImgNav" );
}

void VoxPhanImgNav::track_to_in( Particles particles )
{        
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( particles.size + mh_params->gpu_block_size - 1 ) / mh_params->gpu_block_size;

    VPIN::kernel_device_track_to_in<<<grid, threads>>>( particles.data_d, m_phantom.h_volume->xmin, m_phantom.h_volume->xmax,
                                                        m_phantom.h_volume->ymin, m_phantom.h_volume->ymax,
                                                        m_phantom.h_volume->zmin, m_phantom.h_volume->zmax,
                                                        mh_params->geom_tolerance );
    cuda_error_check ( "Error ", " Kernel_VoxPhanImgNav (track to in)" );
    cudaDeviceSynchronize();

}

void VoxPhanImgNav::track_to_out( Particles particles )
{
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( particles.size + mh_params->gpu_block_size - 1 ) / mh_params->gpu_block_size;

    // DEBUG
    VPIN::kernel_device_track_to_out<<<grid, threads>>> ( particles.data_d, m_phantom.d_volume, m_materials.d_materials,
                                                          m_cross_sections.d_photon_CS, md_params );
    cuda_error_check ( "Error ", " Kernel_VoxPhanImgNav (track to out)" );
    cudaDeviceSynchronize();
}

void VoxPhanImgNav::load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
{   
    m_phantom.load_from_mhd( filename, range_mat_name );
}

void VoxPhanImgNav::initialize(GlobalSimulationParametersData *h_params, GlobalSimulationParametersData *d_params)
{
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanImgNav: missing parameters." );
        exit_simulation();
    }

    // Params
    mh_params = h_params;
    md_params = d_params;

    // Phantom
    m_phantom.set_name( "VoxPhanImgNav" );
    m_phantom.initialize();

    // Material
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize( m_phantom.list_of_materials, h_params );

    // Cross Sections
    m_cross_sections.initialize( m_materials.h_materials, h_params );

    // Some verbose if required
    if ( h_params->display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanImgNav", mem);
    }

}

void VoxPhanImgNav::set_materials( std::string filename )
{
    m_materials_filename = filename;
}

AabbData VoxPhanImgNav::get_bounding_box()
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
