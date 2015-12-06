// GGEMS Copyright (C) 2015

/*!
 * \file point_source.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef POINT_SOURCE_CU
#define POINT_SOURCE_CU

#include "point_source.cuh"

///////// GPU code ////////////////////////////////////////////////////

// Internal function
__host__ __device__ void point_source(ParticlesData particles_data, ui32 id,
                                      f32 px, f32 py, f32 pz, ui8 type,
                                      f64 *spectrumE, f64 *spectrumCDF, ui32 nbins) {

    f32 phi = JKISS32(particles_data, id);
    f32 theta = JKISS32(particles_data, id);

    phi  *= gpu_twopi;
    theta = acosf(1.0f - 2.0f*theta);

    ui32 pos = binary_search(JKISS32(particles_data, id), spectrumCDF, nbins);

    // set photons
    particles_data.E[id] = spectrumE[pos];
    particles_data.dx[id] = cosf(phi)*sinf(theta);
    particles_data.dy[id] = sinf(phi)*sinf(theta);
    particles_data.dz[id] = cosf(theta);
    particles_data.px[id] = px;
    particles_data.py[id] = py;
    particles_data.pz[id] = pz;
    particles_data.tof[id] = 0.0f;
    particles_data.endsimu[id] = PARTICLE_ALIVE;
    particles_data.next_discrete_process[id] = NO_PROCESS;
    particles_data.next_interaction_distance[id] = 0.0;
    particles_data.level[id] = PRIMARY;
    particles_data.pname[id] = type;
    particles_data.geometry_id[id] = 0;
}

// Kernel to create new particles (sources manager)
__global__ void kernel_point_source(ParticlesData particles_data,
                                    f32 px, f32 py, f32 pz, ui8 type,
                                    f64 *spectrumE, f64 *spectrumCDF, ui32 nbins) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles_data.size) return;

    point_source(particles_data, id, px, py, pz, type,
                 spectrumE, spectrumCDF, nbins);

}

///////////////////////////////////////////////////////////////////////

// Constructor
PointSource::PointSource() {
    // Default parameters
    m_px = 0.0f; m_py = 0.0f; m_pz = 0.0f;
    m_nb_of_energy_bins = 0;
    m_spectrumE_h = NULL;
    m_spectrumE_d = NULL;
    m_spectrumCDF_h = NULL;
    m_spectrumCDF_d = NULL;
    m_particle_type = PHOTON;
}

// Destructor
PointSource::~PointSource() {
    free(m_spectrumE_h);
    free(m_spectrumCDF_h);
    cudaFree(m_spectrumE_d);
    cudaFree(m_spectrumCDF_d);
}

// Setting function
void PointSource::set_position(f32 vpx, f32 vpy, f32 vpz) {
    m_px=vpx; m_py=vpy; m_pz=vpz;
}

void PointSource::set_particle_type(std::string pname) {
    if (pname == "photon") {
        m_particle_type = PHOTON;
    } else if (pname == "electron") {
        m_particle_type = ELECTRON;
    } else if (pname == "positron") {
        m_particle_type = POSITRON;
    }
}

void PointSource::set_mono_energy(f32 valE) {
    m_spectrumE_h = (f64*)malloc(sizeof(f64));
    m_spectrumE_h[0] = valE;
    m_spectrumCDF_h = (f64*)malloc(sizeof(f64));
    m_spectrumCDF_h[0] = 1.0;
    m_nb_of_energy_bins = 1;
}

void PointSource::set_energy_spectrum(f64 *valE, f64 *hist, ui32 nb) {

    // Allocation
    m_spectrumE_h = (f64*)malloc(nb*sizeof(f64));
    m_spectrumCDF_h = (f64*)malloc(nb*sizeof(f64));
    m_nb_of_energy_bins = nb;

    // Get the sum
    f64 sum = 0;
    ui32 i = 0;
    while (i<nb) {
        sum += hist[i];
        ++i;
    }
    // Normalize
    i=0; while (i<nb) {
        m_spectrumCDF_h[i] = hist[i] / sum;
        // In the mean time copy energy value
        m_spectrumE_h[i] = valE[i];
        ++i;
    }
    // Get the final CDF
    i=1; while (i<nb) {
        m_spectrumCDF_h[i] += m_spectrumCDF_h[i-1];
        ++i;
    }
    // Watchdog
    m_spectrumCDF_h[nb-1] = 1.0f;
}

// Main function
bool PointSource::m_check_mandatory() {
    if (m_nb_of_energy_bins == 0) return false;
    else return true;
}

void PointSource::initialize(GlobalSimulationParameters params) {

    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("Missing parameters for the point source!");
        exit_simulation();
    }

    // Store global parameters
    m_params = params;

    // Handle GPU device
    if (m_params.data_h.device_target == GPU_DEVICE && m_nb_of_energy_bins > 1) {
        // GPU mem allocation
        HANDLE_ERROR( cudaMalloc((void**) &m_spectrumE_d, m_nb_of_energy_bins*sizeof(f64)) );
        HANDLE_ERROR( cudaMalloc((void**) &m_spectrumCDF_d, m_nb_of_energy_bins*sizeof(f64)) );
        // GPU mem copy
        HANDLE_ERROR( cudaMemcpy(m_spectrumE_d, m_spectrumE_h,
                                 sizeof(f64)*m_nb_of_energy_bins, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(m_spectrumCDF_d, m_spectrumCDF_h,
                                 sizeof(f64)*m_nb_of_energy_bins, cudaMemcpyHostToDevice) );
    }

}

void PointSource::get_primaries_generator(Particles particles) {

    if (m_params.data_h.device_target == CPU_DEVICE) {

        ui32 id=0; while (id<particles.size) {
            point_source(particles.data_h, id, m_px, m_py, m_pz, m_particle_type,
                         m_spectrumE_d, m_spectrumCDF_d, m_nb_of_energy_bins);
            ++id;
        }

    } else if (m_params.data_h.device_target == GPU_DEVICE) {

        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = (particles.size + m_params.data_h.gpu_block_size - 1) / m_params.data_h.gpu_block_size;

        kernel_point_source<<<grid, threads>>>(particles.data_d, m_px, m_py, m_pz, m_particle_type,
                                               m_spectrumE_d, m_spectrumCDF_d, m_nb_of_energy_bins);
        cuda_error_check("Error ", " Kernel_point_source");

    }

}

#endif

















