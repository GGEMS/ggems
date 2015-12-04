// GGEMS Copyright (C) 2015

/*!
 * \file flatpanel_detector.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef FLATPANEL_DETECTOR_CU
#define FLATPANEL_DETECTOR_CU

#include "flatpanel_detector.cuh"

////:: GPU Codes

// Move particles to the voxelized volume
__host__ __device__ void flatpanel_track_to_in(ParticlesData &particles, ObbData obb, f32* projection,
                                               f32 pixel_size_x, f32 pixel_size_y,
                                               ui16 nb_pixel_x, ui16 nb_pixel_y,
                                               ui32 id) {

    // If freeze (not dead), re-activate the current particle
    if (particles.endsimu[id] == PARTICLE_FREEZE) particles.endsimu[id] = PARTICLE_ALIVE;

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

    // TODO TODO TODO

    /*
    f32 dist = hit_ray_AABB(pos, dir, obb.xmin, obb.xmax, obb.ymin, obb.ymax, obb.zmin, obb.zmax);

    // TODO here should use hit_ray_OBB - JB

    // the particle not hitting the voxelized volume
    if (dist == FLT_MAX) {                            // TODO: Don't know why F32_MAX doesn't work...
        particles.endsimu[id] = PARTICLE_FREEZE;
        return;
    } else {
        // Check if the path of the particle cross the volume sufficiently
        f32 cross = dist_overlap_ray_AABB(pos, dir, obb.xmin, obb.xmax, obb.ymin, obb.ymax, obb.zmin, obb.zmax);
        if (cross < EPSILON3) {
            particles.endsimu[id] = PARTICLE_FREEZE;
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add(pos, fxyz_scale(pos, dist+EPSILON6));

    }
    */

    // Calculate pixel id
    ui16 ix = (pos.x+obb.xmin) / pixel_size_x;
    ui16 iy = (pos.z+obb.zmin) / pixel_size_y;

    // DEBUG Check index
    if (ix >= nb_pixel_x || iy >= nb_pixel_y) {
        printf("Pixel index out of FOV %i %i\n", ix, iy);
        return;
    }

    // Drop the complete energy
#ifdef __CUDA_ARCH__
    atomicAdd(&projection[iy*nb_pixel_x + ix], particles.E[id]);
#else
    projection[iy*nb_pixel_x + ix] += particles.E[id];
#endif

    particles.endsimu[id] = PARTICLE_DEAD;

}

// Kernel that move particles to the voxelized volume boundary
__global__ void kernel_flatpanel_track_to_in(ParticlesData particles, ObbData obb, float* projection,
                                             f32 pixel_size_x, f32 pixel_size_y,
                                             ui16 nb_pixel_x, ui16 nb_pixel_y) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    flatpanel_track_to_in(particles, obb, projection, pixel_size_x, pixel_size_y, nb_pixel_x, nb_pixel_y,id);

}


FlatpanelDetector::FlatpanelDetector() {
    m_pixel_size_x = 0;
    m_pixel_size_y = 0;

    m_nb_pixel_x = 0;
    m_nb_pixel_y = 0;
}

////:: Privates

// Copy the phantom to the GPU
void FlatpanelDetector::m_copy_detector_cpu2gpu() {

    m_phantom.volume.data_d.xmin = m_phantom.volume.data_h.xmin;
    m_phantom.volume.data_d.xmax = m_phantom.volume.data_h.xmax;

    m_phantom.volume.data_d.ymin = m_phantom.volume.data_h.ymin;
    m_phantom.volume.data_d.ymax = m_phantom.volume.data_h.ymax;

    m_phantom.volume.data_d.zmin = m_phantom.volume.data_h.zmin;
    m_phantom.volume.data_d.zmax = m_phantom.volume.data_h.zmax;

    m_phantom.volume.data_d.angle = m_phantom.volume.data_h.angle;
    m_phantom.volume.data_d.translate = m_phantom.volume.data_h.translate;
    m_phantom.volume.data_d.center = m_phantom.volume.data_h.center;

    m_phantom.volume.data_d.u = m_phantom.volume.data_h.u;
    m_phantom.volume.data_d.v = m_phantom.volume.data_h.v;
    m_phantom.volume.data_d.w = m_phantom.volume.data_h.w;

    m_phantom.volume.data_d.size = m_phantom.volume.data_h.size;

}

bool FlatpanelDetector::m_check_mandatory() {

    if (m_pixel_size_x == 0 || m_pixel_size_y == 0) {
        return false;
    } else {
        return true;
    }

}

////:: Setting

void FlatpanelDetector::set_width_and_height(f32 w, f32 h) {
    m_phantom.set_size(w, 1.0, h); // Flatpanel xOz with thickness of y=1 mm
}

void FlatpanelDetector::set_pixel_size(f32 sx, f32 sy) {
    m_pixel_size_x = sx;
    m_pixel_size_y = sy;
}

void FlatpanelDetector::set_orbiting_radius(f32 r) {
    m_phantom.translate(0.0, r, 0.0);  // Behind the patient i.e. y-axis positif
}

////:: Main functions

void FlatpanelDetector::track_to_in(Particles particles) {

    if (m_params.data_h.device_target == CPU_DEVICE) {
        ui32 id=0; while (id<particles.size) {
            flatpanel_track_to_in(particles.data_h,  m_phantom.volume.data_h, m_projection_h,
                                  m_pixel_size_x, m_pixel_size_y,
                                  m_nb_pixel_x, m_nb_pixel_y, id);
            ++id;
        }
    } else if (m_params.data_h.device_target == GPU_DEVICE) {

        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = (particles.size + m_params.data_h.gpu_block_size - 1) / m_params.data_h.gpu_block_size;

        kernel_flatpanel_track_to_in<<<grid, threads>>>(particles.data_d, m_phantom.volume.data_d, m_projection_d,
                                                        m_pixel_size_x, m_pixel_size_y,
                                                        m_nb_pixel_x, m_nb_pixel_y);
        cuda_error_check("Error ", " Kernel_Flatpanel (track to in)");

    }


}

void FlatpanelDetector::initialize(GlobalSimulationParameters params) {
    // Check params
    if (!m_check_mandatory()) {
        print_error("FlatpanelDetector: missing parameters.");
        exit_simulation();
    }

    // Params
    m_params = params;

    // Compute some params
    m_nb_pixel_x = (ui16)(m_phantom.volume.data_h.size.x / m_pixel_size_x);
    m_nb_pixel_y = (ui16)(m_phantom.volume.data_h.size.z / m_pixel_size_y); // Flatpanel xOz

    // Init proj
    m_projection_h = (f32*)malloc(m_nb_pixel_x*m_nb_pixel_y*sizeof(f32));
    ui32 i=0; while(i<m_nb_pixel_x*m_nb_pixel_y) {m_projection_h[i] = 0.0; ++i;}

    // Copy data to GPU
    if (m_params.data_h.device_target == GPU_DEVICE) {
        m_copy_detector_cpu2gpu();

        // GPU mem allocation
        HANDLE_ERROR( cudaMalloc((void**) &m_projection_d, m_nb_pixel_x*m_nb_pixel_y*sizeof(f32)) );

        // GPU mem copy
        HANDLE_ERROR( cudaMemcpy(m_projection_d, m_projection_h,
                                 sizeof(f32)*m_nb_pixel_x*m_nb_pixel_y, cudaMemcpyHostToDevice) );
    }

}
























#endif
