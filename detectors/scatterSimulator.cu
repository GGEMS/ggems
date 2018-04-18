


#ifndef SCATTERSIMULATOR_CU
#define SCATTERSIMULATOR_CU

#include "scatterSimulator.cuh"

//// GPU Codes //////////////////////////////////////////////

// This function navigate particle from the phantom to the detector (to in).
__host__ __device__ void voxGridDetector_track_to_in( ParticlesData *particles, ui32 id, float* gridDimensions, int spacing, int vox_x, int vox_y, float* energiesPerVoxel, float* energiesPerVoxel_sq, float* hitsPerVoxel, int voxels_N )
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

    float geom_tolerance = 10.0 *mm; // TO CHECK

    float offset = (spacing * vox_x) * 0.5 * mm;

    // Read current particle's position
    f32xyz pos;
    pos.x = particles->px[ id ];
    pos.y = particles->py[ id ];
    pos.z = particles->pz[ id ];

    // Read current particle's direction
    f32xyz dir;
    dir.x = particles->dx[ id ];
    dir.y = particles->dy[ id ];
    dir.z = particles->dz[ id ];

    // Normalize direction
    float3 dir_n;
    dir_n = fxyz_unit( dir );

    // Read current particle's energy
    f32 E = particles->E[ id ]; // / MeV;
    f32 E_sq = E * E;

    // Contsant index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / spacing;
    ivoxsize.y = 1.0 / spacing;
    ivoxsize.z = 1.0 / spacing;

    // Get detector's voxel index (TMP)
    /*ui32xyzw index_voxel;
    index_voxel.x = ui32( (detectorPos.x + offset) * ivoxsize.x );
    index_voxel.y = ui32( (detectorPos.y + offset) * ivoxsize.y );
    index_voxel.z = ui32( (detectorPos.z + offset) * ivoxsize.z );
    index_voxel.w = index_voxel.z * vox_x * vox_y + index_voxel.y * vox_x + index_voxel.x; // linear index
    f32 vox_det_xmin = (index_voxel.x * spacing - offset) * 5;
    f32 vox_det_ymin = (index_voxel.y * spacing - offset) * 5;
    f32 vox_det_zmin = (index_voxel.z * spacing - offset) * 5;
    f32 vox_det_xmax = (vox_det_xmin + spacing) * 5;
    f32 vox_det_ymax = (vox_det_ymin + spacing) * 5;
    f32 vox_det_zmax = (vox_det_zmin + spacing) * 5;*/

    // Navigation inside voxelized structure
    bool particle_in_grid = true;
    while( particle_in_grid )
    {
	// Get the particle's voxel position
        ui32xyzw index_phantom;
	index_phantom.x = ui32( (pos.x + offset) * ivoxsize.x );
	index_phantom.y = ui32( (pos.y + offset) * ivoxsize.y );
	index_phantom.z = ui32( (pos.z + offset) * ivoxsize.z );
	index_phantom.w = index_phantom.z * vox_x * vox_y + index_phantom.y * vox_x + index_phantom.x; // linear index

	// Save informations about particle in grid	
	if( index_phantom.w <= voxels_N && index_phantom.w >= 0 )
	{
		ggems_atomic_add( energiesPerVoxel, index_phantom.w, E );
		ggems_atomic_add( energiesPerVoxel_sq, index_phantom.w, E_sq );
		ggems_atomic_add( hitsPerVoxel, index_phantom.w, 1 );
	}

	//// Get the next distance boundary volume (voxel) /////////////////////////////////

	// get voxel params
	f32 vox_xmin = index_phantom.x * spacing - offset;
	f32 vox_ymin = index_phantom.y * spacing - offset;
	f32 vox_zmin = index_phantom.z * spacing - offset;
	f32 vox_xmax = vox_xmin + spacing;
	f32 vox_ymax = vox_ymin + spacing;
	f32 vox_zmax = vox_zmin + spacing;

	// get a safety position for the particle within this voxel (sometimes a particle can be right between two voxels)
	pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax, vox_ymin, vox_ymax, vox_zmin, vox_zmax, geom_tolerance );


	// compute the next distance boundary
	f32 boundary_distance = hit_ray_AABB( pos, dir, vox_xmin, vox_xmax, vox_ymin, vox_ymax, vox_zmin, vox_zmax );

	//// Move particle //////////////////////////////////////////////////////
	// get the new position
	pos = fxyz_add( pos, fxyz_scale( dir, boundary_distance ) );

	// get safety position (outside the current voxel)
	pos = transport_get_safety_outside_AABB( pos, vox_xmin, vox_xmax, vox_ymin, vox_ymax, vox_zmin, vox_zmax, geom_tolerance );   

	// Stop simulation if out of the phantom
	if ( !test_point_AABB_with_tolerance( pos, gridDimensions[ 0 ], gridDimensions[ 1 ], gridDimensions[ 2 ], gridDimensions[ 3 ], gridDimensions[ 4], gridDimensions[ 5 ], geom_tolerance ) )
	{          
		particle_in_grid = false;
	}

    }

}

// This function navigate particle within the detector until escaping (to out)
// Most of the time particle navigation is not required into detector. Here we not using it.

//__host__ __device__ void dummy_detector_track_to_out( ParticlesData &particles, ui32 id ) {}


// Digitizer record and process data into the detector. For example in CT imaging the digitizer will compute
// the number of particle per pixel.
__host__ __device__ void voxGridDetector_digitizer( ParticlesData *particles, ui32 id )
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
__global__ void kernel_voxGridDetector_track_to_in( ParticlesData *particles, float* gridDimensions, int spacing, int vox_x, int vox_y, float* energiesPerVoxel, float* energiesPerVoxel_sq, float* hitsPerVoxel, int voxels_N )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles->size) return;

    voxGridDetector_track_to_in( particles, id, gridDimensions, spacing, vox_x, vox_y, energiesPerVoxel, energiesPerVoxel_sq, hitsPerVoxel, voxels_N);
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
__global__ void kernel_voxGridDetector_digitizer( ParticlesData *particles )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles->size) return;

    voxGridDetector_digitizer( particles, id );
}

void write_voxGrid_data( std::string filename, float* energiesPerVox, float* energiesPerVox_squared, float* hitsPerVox, int size )
{
    std::ofstream outputFile;

    outputFile.open( filename.c_str(), std::ios::out | std::ios::trunc );	

    // Write data per voxel to a file
    for( int i = 0; i < size; i++ )
    {
	outputFile << i << "\t" << energiesPerVox[ i ] << "\t" << energiesPerVox_squared[ i ] << "\t" << hitsPerVox[ i ] << "\n";
    }

    outputFile.close();   
}

//// VoxGridDetector class ///////////////////////////////////////////////

VoxGridDetector::VoxGridDetector() : GGEMSDetector(),
                                      m_width( 0.0f ),
                                      m_height( 0.0f ),
                                      m_depth( 0.0f )
{
        // Default values
    set_name( "voxelizedGrid_detector" );

    m_grid_center = make_float3(0.0, 0.0, 0.0);
    m_voxelizedGriddims_h = new float[6];
    m_voxelizedGriddims_h[0] = 0.0;
    m_voxelizedGriddims_h[1] = 0.0;
    m_voxelizedGriddims_h[2] = 0.0;
    m_voxelizedGriddims_h[3] = 0.0;
    m_voxelizedGriddims_h[4] = 0.0;
    m_voxelizedGriddims_h[5] = 0.0;

    GGcout << "Voxelized Grid Detector created !" << GGendl;
}

VoxGridDetector::~VoxGridDetector() {}

//============ Setting functions ==========================

void VoxGridDetector::set_dimension( f32 width, f32 height, f32 depth )
{
    m_width = width;
    m_height = height;
    m_depth = depth;

    GGcout<<"*******Dimension set " << m_width <<" "<<m_height <<" " << m_depth << GGendl;
}

void VoxGridDetector::set_source_position( f32 center_x, f32 center_y, f32 center_z )
{
    m_source_p.x = center_x;
    m_source_p.y = center_y;
    m_source_p.z = center_z;

    //! Compute detector's position
    f32 dist2Detector = 500.0;
    f32xyz srcDir = fxyz_unit( m_source_p );
    srcDir = fxyz_scale(srcDir, -1); // inverse direction
    f32xyz DetectorPos = fxyz_scale(srcDir, dist2Detector);
    //m_detectorPos = make_float3(DetectorPos.x, DetectorPos.y, DetectorPos.z);
    m_detectorPos = make_float3(500.0, 0.0, 500.0);

    GGcout<<"*******Source pos set " << GGendl;
}

void VoxGridDetector::set_grid_center( f32 x, f32 y, f32 z )
{
    // Grid's center
    m_grid_center = make_float3(x, y, z);   
    
    // Compute the grid's limits
    m_voxelizedGriddims_h[0] = m_grid_center.x - (m_width / 2); //xmin
    m_voxelizedGriddims_h[1] = m_grid_center.x + (m_width / 2); //xmax
    m_voxelizedGriddims_h[2] = m_grid_center.y - (m_height / 2); //ymin
    m_voxelizedGriddims_h[3] = m_grid_center.y + (m_height / 2); //ymax
    m_voxelizedGriddims_h[4] = m_grid_center.z - (m_depth / 2); //zmin
    m_voxelizedGriddims_h[5] = m_grid_center.z + (m_depth / 2); //zmax

    GGcout<<"*******Voxelized Grid's Position set " << GGendl;
    GGcout<<"*******xmin " << m_voxelizedGriddims_h[0] << " xmax " << m_voxelizedGriddims_h[1] << GGendl;
    GGcout<<"*******ymin " << m_voxelizedGriddims_h[2] << " ymax " << m_voxelizedGriddims_h[3] << GGendl;
    GGcout<<"*******zmin " << m_voxelizedGriddims_h[4] << " zmax " << m_voxelizedGriddims_h[5] << GGendl;
}

void VoxGridDetector::buildVoxelGrid()
{
    m_voxels_x = m_width / m_voxelSpacing;
    m_voxels_y = m_height / m_voxelSpacing;
    m_voxels_z = m_depth / m_voxelSpacing;

    m_nb_of_Voxels = m_voxels_x * m_voxels_y * m_voxels_z;

    GGcout << "VOXELS: " << m_nb_of_Voxels << "\t" << m_voxels_x << "x" << m_voxels_y << "x" << m_voxels_z << GGendl;

    // Initialization of host arrays
    m_energies_per_Voxel_h = new float[ m_nb_of_Voxels ];
    m_energies_per_Voxel_squared_h = new float[ m_nb_of_Voxels ];
    m_hits_per_Voxel_h = new float[ m_nb_of_Voxels ];
    m_uncertainty_per_Voxel = new float[ m_nb_of_Voxels ];

    GGcout<<"*******Voxelized Grid was BUILT " << GGendl;
}

//============ Some functions ============================

void VoxGridDetector::setDataOutputPath( std::string filename )
{    
    m_dataOutputPath = filename;
}

// Export data
void VoxGridDetector::save_data( )
{
    std::string filename_stats = m_dataOutputPath + "_ScatterRad.txt";

    // Copy data from GPU to CPU
    //HANDLE_ERROR( cudaMemcpy(m_particlesPos_h, m_particlesPos_d, NB_OF_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost) ); // For debugging
    HANDLE_ERROR( cudaMemcpy(m_energies_per_Voxel_h, m_energies_per_Voxel_d, m_nb_of_Voxels * sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(m_energies_per_Voxel_squared_h, m_energies_per_Voxel_squared_d, m_nb_of_Voxels * sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(m_hits_per_Voxel_h, m_hits_per_Voxel_d, m_nb_of_Voxels * sizeof(float), cudaMemcpyDeviceToHost) );

    // Save results
    write_voxGrid_data( filename_stats, m_energies_per_Voxel_h, m_energies_per_Voxel_squared_h, m_hits_per_Voxel_h, m_nb_of_Voxels );

    GGcout << "**/*/* Voxelized grid results saved in "<< filename_stats << GGendl;
}

//============ Mandatory functions =======================

// Check if everything is ok to initialize this detector
bool VoxGridDetector::m_check_mandatory()
{
    if ( m_width == 0.0 || m_height == 0.0 || m_depth == 0.0 || m_voxelSpacing == 0.0 ) return false;
    else return true;
}

// This function is mandatory and called by GGEMS to initialize and load all
// necessary data on the graphic card
void VoxGridDetector::initialize( GlobalSimulationParametersData *params )
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
    m_nb_of_particles = mh_params->nb_of_particles;

    // Initialization of the voxelized structure
    this->buildVoxelGrid();

    // Handle GPU device if needed. Here nothing is load to the GPU (simple template). But
    // in case of the use of data on the GPU you should allocated and transfered here.

    // GPU mem allocation
    // Set to 0 all the arrays
    HANDLE_ERROR( cudaMalloc( (void**)&m_energies_per_Voxel_d, m_nb_of_Voxels * sizeof(float)));
    HANDLE_ERROR( cudaMemset(m_energies_per_Voxel_d, 0, m_nb_of_Voxels * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&m_energies_per_Voxel_squared_d, m_nb_of_Voxels * sizeof(float)));
    HANDLE_ERROR( cudaMemset(m_energies_per_Voxel_squared_d, 0, m_nb_of_Voxels * sizeof(float)));
    HANDLE_ERROR( cudaMalloc( (void**)&m_hits_per_Voxel_d, m_nb_of_Voxels * sizeof(float)));
    HANDLE_ERROR( cudaMemset(m_hits_per_Voxel_d, 0, m_nb_of_Voxels * sizeof(float)));

    // GPU mem copy
    int N = 6; // 6 dimensions' array
    HANDLE_ERROR( cudaMalloc( (void**)&m_voxelizedGriddims_d, N * sizeof(float)));
    HANDLE_ERROR( cudaMemcpy( m_voxelizedGriddims_d, m_voxelizedGriddims_h, N * sizeof( float ), cudaMemcpyHostToDevice ) );

    GGcout << "Voxelized Grid detector correctly initialized" << GGendl;
}

// Mandatory function, that handle track_to_in
void VoxGridDetector::track_to_in( ParticlesData *d_particles )
{

    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 )
             / mh_params->gpu_block_size;


    // Execution og GPU Kernel function
    kernel_voxGridDetector_track_to_in<<<grid, threads>>>( d_particles, m_voxelizedGriddims_d, m_voxelSpacing, m_voxels_y, m_voxels_z, m_energies_per_Voxel_d, m_energies_per_Voxel_squared_d, m_hits_per_Voxel_d, m_nb_of_Voxels );
    cuda_error_check("Error ", " Kernel_template_detector (track to in)");
    cudaThreadSynchronize();

}

// If navigation within the detector is required te track_to_out function should be
// equivalent to the track_to_in function. Here there is no navigation. However, this function
// is mandatory, and must be defined
void VoxGridDetector::track_to_out( ParticlesData *d_particles ) {}

// Same mandatory function to drive the digitizer function between CPU and GPU
void VoxGridDetector::digitizer( ParticlesData *d_particles )
{
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 )
             / mh_params->gpu_block_size;

    kernel_voxGridDetector_digitizer<<<grid, threads>>>( d_particles );
    cuda_error_check("Error ", " Kernel_template_detector (digitizer)");
    cudaThreadSynchronize();
}



#endif
