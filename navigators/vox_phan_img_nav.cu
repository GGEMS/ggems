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

// Move particles to the voxelized volume
__host__ __device__ void VPIN::track_to_in ( ParticlesData &particles, f32 xmin, f32 xmax,
        f32 ymin, f32 ymax, f32 zmin, f32 zmax,
        ui32 id )
{

    // Read position
    f64xyz pos;
    pos.x = particles.px[id];
    pos.y = particles.py[id];
    pos.z = particles.pz[id];

    // Read direction
    f64xyz dir;
    dir.x = particles.dx[id];
    dir.y = particles.dy[id];
    dir.z = particles.dz[id];

    f32 dist = hit_ray_AABB ( pos, dir, xmin, xmax, ymin, ymax, zmin, zmax );

    // the particle not hitting the voxelized volume
    if ( dist == FLT_MAX )                            // TODO: Don't know why F32_MAX doesn't work...
    {
        particles.endsimu[id] = PARTICLE_FREEZE;
        return;
    }
    else
    {
        // Check if the path of the particle cross the volume sufficiently
        f32 cross = dist_overlap_ray_AABB ( pos, dir, xmin, xmax, ymin, ymax, zmin, zmax );
        if ( cross < EPSILON3 )
        {
            particles.endsimu[id] = PARTICLE_FREEZE;
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add ( pos, fxyz_scale ( pos, dist+EPSILON6 ) );

        // TODO update tof
        // ...
    }

    // set photons
    particles.px[id] = pos.x;
    particles.py[id] = pos.y;
    particles.pz[id] = pos.z;

}


__host__ __device__ void VPIN::track_to_out ( ParticlesData &particles,
        VoxVolumeData vol,
        MaterialsTable materials,
        PhotonCrossSectionTable photon_CS_table,
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
    ui16xyzw index_phantom;
    index_phantom.x = ui16 ( ( pos.x+vol.off_x ) * ivoxsize.x );
    index_phantom.y = ui16 ( ( pos.y+vol.off_y ) * ivoxsize.y );
    index_phantom.z = ui16 ( ( pos.z+vol.off_z ) * ivoxsize.z );
    index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                      + index_phantom.y*vol.nb_vox_x
                      + index_phantom.x; // linear index

    // Get the material that compose this volume
    ui16 mat_id = vol.values[index_phantom.w];

    //// Find next discrete interaction ///////////////////////////////////////

    photon_get_next_interaction ( particles, parameters, photon_CS_table, mat_id, part_id );

    f32 next_interaction_distance = particles.next_interaction_distance[part_id];
    ui8 next_discrete_process = particles.next_discrete_process[part_id];

    //// Get the next distance boundary volume /////////////////////////////////

    f32 vox_xmin = index_phantom.x*vol.spacing_x;
    f32 vox_ymin = index_phantom.y*vol.spacing_y;
    f32 vox_zmin = index_phantom.z*vol.spacing_z;
    f32 vox_xmax = vox_xmin + vol.spacing_x;
    f32 vox_ymax = vox_ymin + vol.spacing_y;
    f32 vox_zmax = vox_zmin + vol.spacing_z;

    f32 boundary_distance = hit_ray_AABB ( pos, dir, vox_xmin, vox_xmax,
                                           vox_ymin, vox_ymax, vox_zmin, vox_zmax );

    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + EPSILON3; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // Update TOF - TODO
    //particles.tof[part_id] += c_light * next_interaction_distance;

    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB ( pos, vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax ) )
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

        //// Here e- are not tracked, and lost energy not drop

    }

    //// Energy cut
    if ( particles.E[part_id] <= materials.electron_energy_cut[mat_id] )
    {
        particles.endsimu[part_id] = PARTICLE_DEAD;
        return;
    }

}

// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPIN::kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
        f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    VPIN::track_to_in ( particles, xmin, xmax, ymin, ymax, zmin, zmax, id );

}

// Host Kernel that move particles to the voxelized volume boundary
void VPIN::kernel_host_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                     f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 id )
{

    VPIN::track_to_in ( particles, xmin, xmax, ymin, ymax, zmin, zmax, id );

}

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPIN::kernel_device_track_to_out ( ParticlesData particles,
        VoxVolumeData vol,
        MaterialsTable materials,
        PhotonCrossSectionTable photon_CS_table,
        GlobalSimulationParametersData parameters )
{

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    // Stepping loop
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {
        VPIN::track_to_out ( particles, vol, materials, photon_CS_table, parameters, id );
    }

}

// Host kernel that track particles within the voxelized volume until boundary
void VPIN::kernel_host_track_to_out ( ParticlesData particles,
                                      VoxVolumeData vol,
                                      MaterialsTable materials,
                                      PhotonCrossSectionTable photon_CS_table,
                                      GlobalSimulationParametersData parameters, ui32 id )
{

    // Stepping loop
    while ( particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE )
    {
        VPIN::track_to_out ( particles, vol, materials, photon_CS_table, parameters, id );
    }
}

////:: Privates

bool VoxPhanImgNav::m_check_mandatory()
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

void VoxPhanImgNav::track_to_in ( Particles particles )
{

    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id=0;
        while ( id<particles.size )
        {
            VPIN::kernel_host_track_to_in ( particles.data_h, m_phantom.data_h.xmin, m_phantom.data_h.xmax,
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

        VPIN::kernel_device_track_to_in<<<grid, threads>>> ( particles.data_d, m_phantom.data_h.xmin, m_phantom.data_h.xmax,
                m_phantom.data_h.ymin, m_phantom.data_h.ymax,
                m_phantom.data_h.zmin, m_phantom.data_h.zmax );
        cuda_error_check ( "Error ", " Kernel_VoxPhanImgNav (track to in)" );

    }


}

void VoxPhanImgNav::track_to_out ( Particles particles )
{

    if ( m_params.data_h.device_target == CPU_DEVICE )
    {

        ui32 id=0;
        while ( id<particles.size )
        {

            VPIN::kernel_host_track_to_out ( particles.data_h, m_phantom.data_h,
                                             m_materials.data_h, m_cross_sections.photon_CS.data_h, m_params.data_h, id );
            ++id;
        }
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {

        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

        VPIN::kernel_device_track_to_out<<<grid, threads>>> ( particles.data_d, m_phantom.data_d, m_materials.data_d,
                m_cross_sections.photon_CS.data_d, m_params.data_d );
        cuda_error_check ( "Error ", " Kernel_VoxPhanImgNav (track to out)" );

    }

}

void VoxPhanImgNav::load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
{
    m_phantom.load_from_mhd ( filename, range_mat_name );
}

void VoxPhanImgNav::initialize ( GlobalSimulationParameters params )
{
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanImgNav: missing parameters." );
        exit_simulation();
    }

    // Params
    m_params = params;

    // Phantom
    m_phantom.set_name ( "VoxPhanImgNav" );
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

    m_materials.initialize ( m_phantom.list_of_materials, params );

    // Cross Sections
    m_cross_sections.initialize ( m_materials, params );
}

void VoxPhanImgNav::set_elements( std::string filename )
{
  m_elements_filename = filename;
}

void VoxPhanImgNav::set_materials( std::string filename )
{
  m_materials_filename = filename;
}





















#endif
