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
__host__ __device__ void vox_phan_track_to_in(ParticleStack &particles, f32 xmin, f32 xmax,
                                              f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                                              ui32 id) {

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

    f32 dist = hit_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax);

    // the particle not hitting the voxelized volume
    if (dist == FLT_MAX) {                            // TODO: Don't know why F32_MAX doesn't work...
        particles.endsimu[id] = PARTICLE_FREEZE;
        return;
    } else {
        // Check if the path of the particle cross the volume sufficiently
        f32 cross = dist_overlap_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax);
        if (cross < EPSILON3) {
            particles.endsimu[id] = PARTICLE_FREEZE;
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add(pos, fxyz_scale(pos, dist+EPSILON6));

        // TODO update tof
        // ...
    }

    // set photons
    particles.px[id] = pos.x;
    particles.py[id] = pos.y;
    particles.pz[id] = pos.z;

}


__host__ __device__ void vox_phan_track_to_out(ParticleStack &particles,
                                               VoxVolume vol,
                                               MaterialsTable materials,
                                               PhotonCrossSectionTable photon_CS_table,
                                               GlobalSimulationParameters parameters,
                                               ui32 part_id) {


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
    index_phantom.x = ui16( (pos.x+vol.org_x) * ivoxsize.x );
    index_phantom.y = ui16( (pos.y+vol.org_y) * ivoxsize.y );
    index_phantom.z = ui16( (pos.z+vol.org_z) * ivoxsize.z );
    index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                      + index_phantom.y*vol.nb_vox_x
                      + index_phantom.x; // linear index

    // Get the material that compose this volume
    ui16 mat_id = vol.data[index_phantom.w];

    //// Find next discrete interaction ///////////////////////////////////////

    photon_get_next_interaction(particles, parameters, photon_CS_table, mat_id, part_id);

    f32 next_interaction_distance = particles.next_interaction_distance[part_id];
    ui8 next_discrete_process = particles.next_discrete_process[part_id];

    //// Get the next distance boundary volume /////////////////////////////////

    f32 vox_xmin = index_phantom.x*vol.spacing_x;
    f32 vox_ymin = index_phantom.y*vol.spacing_y;
    f32 vox_zmin = index_phantom.z*vol.spacing_z;
    f32 vox_xmax = vox_xmin + vol.spacing_x;
    f32 vox_ymax = vox_ymin + vol.spacing_y;
    f32 vox_zmax = vox_zmin + vol.spacing_z;

    f32 boundary_distance = hit_ray_AABB(pos, dir, vox_xmin, vox_xmax,
                                         vox_ymin, vox_ymax, vox_zmin, vox_zmax);

    if (boundary_distance <= next_interaction_distance) {
        next_interaction_distance = boundary_distance + EPSILON3; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    pos = fxyz_add(pos, fxyz_scale(dir, next_interaction_distance));

    // Update TOF - TODO
    //particles.tof[part_id] += c_light * next_interaction_distance;

    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if (!test_point_AABB(pos, vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax)) {
        particles.endsimu[part_id] = PARTICLE_FREEZE;
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    if (next_discrete_process != GEOMETRY_BOUNDARY) {
        // Resolve discrete process
        SecParticle electron = photon_resolve_discrete_process(particles, parameters, photon_CS_table,
                                                               materials, mat_id, part_id);

        //// Here e- are not tracked, and lost energy not drop

    }

    //// Energy cut
    if (particles.E[part_id] <= materials.electron_energy_cut[mat_id]) {
        particles.endsimu[part_id] = PARTICLE_DEAD;
        return;
    }

}

// Kernel that move particles to the voxelized volume boundary
__global__ void kernel_vox_phan_track_to_in(ParticleStack particles, f32 xmin, f32 xmax,
                                            f32 ymin, f32 ymax, f32 zmin, f32 zmax) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    vox_phan_track_to_in(particles, xmin, xmax, ymin, ymax, zmin, zmax, id);

}

// Kernel that track particles within the voxelized volume until boundary
__global__ void kernel_vox_phan_track_to_out(ParticleStack particles,
                                             VoxVolume vol,
                                             MaterialsTable materials,
                                             PhotonCrossSectionTable photon_CS_table,
                                             GlobalSimulationParameters parameters) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    vox_phan_track_to_out(particles, vol, materials, photon_CS_table, parameters, id);

}

////:: Privates

// Copy the global simulation parameters to the GPU
void VoxPhanImgNav::m_copy_parameters_cpu2gpu() {

    // Mem allocation
    HANDLE_ERROR( cudaMalloc((void**) &m_params_d.physics_list, NB_PROCESSES*sizeof(bool)) );
    HANDLE_ERROR( cudaMalloc((void**) &m_params_d.secondaries_list, NB_PARTICLES*sizeof(bool)) );

    // Copy data
    HANDLE_ERROR( cudaMemcpy(m_params_d.physics_list, m_params_h.physics_list,
                         sizeof(bool)*NB_PROCESSES, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(m_params_d.secondaries_list, m_params_h.secondaries_list,
                         sizeof(bool)*NB_PARTICLES, cudaMemcpyHostToDevice) );

    m_params_d.photon_cut = m_params_h.photon_cut;
    m_params_d.electron_cut = m_params_h.electron_cut;

    m_params_d.nb_of_particles = m_params_h.nb_of_particles;
    m_params_d.size_of_particles_batch = m_params_h.size_of_particles_batch;
    m_params_d.nb_of_batches = m_params_h.nb_of_batches;

    m_params_d.device_target = m_params_h.device_target;
    m_params_d.gpu_id = m_params_h.gpu_id;
    m_params_d.gpu_block_size = m_params_h.gpu_block_size;

    m_params_d.time = m_params_h.time;
    m_params_d.seed = m_params_h.seed;

    m_params_d.display_run_time = m_params_h.display_run_time;
    m_params_d.display_memory_usage = m_params_h.display_memory_usage;

    m_params_d.cs_table_nbins = m_params_h.cs_table_nbins;
    m_params_d.cs_table_min_E = m_params_h.cs_table_min_E;
    m_params_d.cs_table_max_E = m_params_h.cs_table_max_E;
}

// Copy the phantom to the GPU
void VoxPhanImgNav::m_copy_phantom_cpu2gpu() {

    // Mem allocation
    HANDLE_ERROR( cudaMalloc((void**) &m_vox_vol_d.data, phantom.volume.number_of_voxels*sizeof(ui16)) );
    // Copy data
    HANDLE_ERROR( cudaMemcpy(m_vox_vol_d.data, phantom.volume.data,
                  phantom.volume.number_of_voxels*sizeof(ui16), cudaMemcpyHostToDevice) );

    m_vox_vol_d.nb_vox_x = phantom.volume.nb_vox_x;
    m_vox_vol_d.nb_vox_y = phantom.volume.nb_vox_y;
    m_vox_vol_d.nb_vox_z = phantom.volume.nb_vox_z;

    m_vox_vol_d.spacing_x = phantom.volume.spacing_x;
    m_vox_vol_d.spacing_y = phantom.volume.spacing_y;
    m_vox_vol_d.spacing_z = phantom.volume.spacing_z;

    m_vox_vol_d.org_x = phantom.volume.org_x;
    m_vox_vol_d.org_y = phantom.volume.org_y;
    m_vox_vol_d.org_z = phantom.volume.org_z;

    m_vox_vol_d.number_of_voxels = phantom.volume.number_of_voxels;
}

bool VoxPhanImgNav::m_check_mandatory() {

    if (phantom.volume.nb_vox_x == 0 || phantom.volume.nb_vox_y == 0 || phantom.volume.nb_vox_z == 0 ||
        phantom.volume.spacing_x == 0 || phantom.volume.spacing_y == 0 || phantom.volume.spacing_z == 0 ||
        phantom.list_of_materials.size() == 0) {
        return false;
    } else {
        return true;
    }

}

////:: Main functions

void VoxPhanImgNav::track_to_in(ParticleStack &particles_h, ParticleStack &particles_d) {

    if (m_params_h.device_target == CPU_DEVICE) {
        ui32 id=0; while (id<particles_h.size) {
            vox_phan_track_to_in(particles_h, phantom.volume.xmin, phantom.volume.xmax,
                                              phantom.volume.ymin, phantom.volume.ymax,
                                              phantom.volume.zmin, phantom.volume.zmax,
                                              id);
            ++id;
        }
    } else if (m_params_h.device_target == GPU_DEVICE) {

        dim3 threads, grid;
        threads.x = m_params_h.gpu_block_size;
        grid.x = (particles_d.size + m_params_h.gpu_block_size - 1) / m_params_h.gpu_block_size;

        kernel_vox_phan_track_to_in<<<grid, threads>>>(particles_d, m_vox_vol_d.xmin, m_vox_vol_d.xmax,
                                                                    m_vox_vol_d.ymin, m_vox_vol_d.ymax,
                                                                    m_vox_vol_d.zmin, m_vox_vol_d.zmax);
        cuda_error_check("Error ", " Kernel_VoxPhanImgNav (track to in)");

    }


}

void VoxPhanImgNav::track_to_out(ParticleStack &particles_h, ParticleStack &particles_d,
                                 MaterialsTable materials_h, MaterialsTable materials_d,
                                 PhotonCrossSectionTable photon_CS_table_h, PhotonCrossSectionTable photon_CS_table_d) {

    if (m_params_h.device_target == CPU_DEVICE) {
        ui32 id=0; while (id<particles_h.size) {
            vox_phan_track_to_out(particles_h, phantom.volume, materials_h, photon_CS_table_h, m_params_h, id);
            ++id;
        }
    } else if (m_params_h.device_target == GPU_DEVICE) {

        dim3 threads, grid;
        threads.x = m_params_h.gpu_block_size;
        grid.x = (particles_d.size + m_params_h.gpu_block_size - 1) / m_params_h.gpu_block_size;

        kernel_vox_phan_track_to_out<<<grid, threads>>>(particles_d, m_vox_vol_d, materials_d,
                                                        photon_CS_table_d, m_params_d);
        cuda_error_check("Error ", " Kernel_VoxPhanImgNav (track to out)");

    }

}

void VoxPhanImgNav::initialize(GlobalSimulationParameters params) {
    // Check params
    if (!m_check_mandatory()) {
        print_error("VoxPhanImgNav: missing parameters.");
        exit_simulation();
    }

    // Params
    m_params_h = params;

    // Phantom name
    phantom.set_name("VoxPhanImgNav");

    // Copy data to GPU
    if (m_params_h.device_target == GPU_DEVICE) {
        m_copy_parameters_cpu2gpu();
        m_copy_phantom_cpu2gpu();
    }

}

// Get list of materials
std::vector<std::string> VoxPhanImgNav::get_materials_list() {
    return phantom.list_of_materials;
}

// Get data that contains materials index
ui16* VoxPhanImgNav::get_data_materials_indices() {
    return phantom.volume.data;
}

// Get the size of data
ui32 VoxPhanImgNav::get_data_size() {
    return phantom.volume.number_of_voxels;
}
























#endif
