// GGEMS Copyright (C) 2015

/*!
 * \file template_detector.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 02/03/2016
 *
 *
 *
 */

#ifndef TEMPLATE_DETECTOR_CU
#define TEMPLATE_DETECTOR_CU

#include "template_detector.cuh"

//// GPU Codes //////////////////////////////////////////////

// This function navigate particle from the phantom to the detector (to in).
__host__ __device__ void template_detector_track_to_in( ParticlesData *particles, ui32 id )
{
    // If freeze (not dead), re-activate the current particle
    if( particles->status[ id ] == PARTICLE_FREEZE )
    {
        particles->status[ id ] = PARTICLE_ALIVE;
    }
    else if ( particles->status[ id ] == PARTICLE_DEAD )
    {
        return;
    }

/*
    // Read position
    f32xyz pos;
    pos.x = particles->px[ id ];
    pos.y = particles->py[ id ];
    pos.z = particles->pz[ id ];

    // Read direction
    f32xyz dir;
    dir.x = particles->dx[ id ];
    dir.y = particles->dy[ id ];
    dir.z = particles->dz[ id ];

    // Do some raytracing function to map particle on the detector boundary

    // Save particle position
    particles->px[ id ] = pos.x;
    particles->py[ id ] = pos.y;
    particles->pz[ id ] = pos.z;

    // ...
    // ...
*/

}

// This function navigate particle within the detector until escaping (to out)
// Most of the time particle navigation is not required into detector. Here we not using it.

//__host__ __device__ void dummy_detector_track_to_out( ParticlesData &particles, ui32 id ) {}


// Digitizer record and process data into the detector. For example in CT imaging the digitizer will compute
// the number of particle per pixel.
__host__ __device__ void template_detector_digitizer( ParticlesData *particles, ui32 id )
{
    // If freeze or dead, quit
    if( particles->status[ id ] == PARTICLE_FREEZE || particles->status[ id ] == PARTICLE_DEAD )
    {
        return;
    }

/*
    // Read position
    f32xyz pos;
    pos.x = particles->px[ id ];
    pos.y = particles->py[ id ];
    pos.z = particles->pz[ id ];

    // Do some processing
*/

}

// Kernel that launch the function track_to_in on GPU
__global__ void kernel_template_detector_track_to_in( ParticlesData *particles )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles->size) return;

    template_detector_track_to_in( particles, id);
}

// If navigation within the detector is required this function must be used
// Kernel that launch the function track_to_in on GPU
//__global__ void kernel_dummy_detector_track_to_out( ParticlesData *particles )
//{
//    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
//    if (id >= particles->size) return;

//    template_detector_track_to_out( particles, id);
//}

// Kernel that launch digitizer on GPU
__global__ void kernel_template_detector_digitizer( ParticlesData *particles )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles->size) return;

    template_detector_digitizer( particles, id );
}

//// TemplateDetector class ///////////////////////////////////////////////

TemplateDetector::TemplateDetector() : GGEMSDetector(),
                                      m_width( 0.0f ),
                                      m_height( 0.0f ),
                                      m_depth( 0.0f )
{
    set_name( "template_detector" );
}

TemplateDetector::~TemplateDetector() {}

//============ Setting function ==========================

void TemplateDetector::set_dimension( f32 width, f32 height, f32 depth )
{
    m_width = width;
    m_height = height;
    m_depth = depth;
}

//============ Some functions ============================

// Export data
void TemplateDetector::save_data( std::string filename )
{
    // If need copy data from GPU to CPU

    // Then export data
    GGcout << "Export data, filename: " << filename << GGendl;
}

//============ Mandatory functions =======================

// Check if everything is ok to initialize this detector
bool TemplateDetector::m_check_mandatory()
{
    if ( m_width == 0.0 || m_height == 0.0 || m_depth == 0.0 ) return false;
    else return true;
}

// This function is mandatory and called by GGEMS to initialize and load all
// necessary data on the graphic card
void TemplateDetector::initialize( GlobalSimulationParametersData *params )
{
    // Check the parameters
    if( !m_check_mandatory() )
    {
        GGcerr << "Template detector was not set properly!" << GGendl;
        exit_simulation();
    }

    // Store global parameters: params are provided by GGEMS and are used to
    // know different information about the simulation. For example if the targeted
    // device is a CPU or a GPU.
    mh_params = params;

    // Handle GPU device if needed. Here nothing is load to the GPU (simple template). But
    // in case of the use of data on the GPU you should allocated and transfered here.

    // GPU mem allocation
    //HANDLE_ERROR( cudaMalloc( (void**)&data_d, N * sizeof( f32 ) ) );

    // GPU mem copy
    //HANDLE_ERROR( cudaMemcpy( data_d, data_h, N * sizeof( f32 ), cudaMemcpyHostToDevice ) );

}

// Mandatory function, that handle track_to_in
void TemplateDetector::track_to_in( ParticlesData *d_particles )
{

    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 )
             / mh_params->gpu_block_size;
    //                                                       pass device data (meaning from GPU)
    kernel_template_detector_track_to_in<<<grid, threads>>>( d_particles );
    cuda_error_check("Error ", " Kernel_template_detector (track to in)");
    cudaThreadSynchronize();

}

// If navigation within the detector is required te track_to_out function should be
// equivalent to the track_to_in function. Here there is no navigation. However, this function
// is mandatory, and must be defined
void TemplateDetector::track_to_out( ParticlesData *d_particles ) {}

// Same mandatory function to drive the digitizer function between CPU and GPU
void TemplateDetector::digitizer( ParticlesData *d_particles )
{
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 )
             / mh_params->gpu_block_size;

    kernel_template_detector_digitizer<<<grid, threads>>>( d_particles );
    cuda_error_check("Error ", " Kernel_template_detector (digitizer)");
    cudaThreadSynchronize();
}



#endif
