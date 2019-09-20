


#ifndef SCATTERSIMULATOR_CU
#define SCATTERSIMULATOR_CU

#include "scatterSimulator.cuh"

//// GPU Codes //////////////////////////////////////////////


// This function navigate particle from the phantom to the detector (to in).
__host__ __device__ void scatterSimulator_track_to_in( ParticlesData *particles, ui32 id, float* gridDimensions, int spacing, int vox_x, int vox_y, float* energiesPerVoxel, float* energiesPerVoxel_sq, float* hitsPerVoxel, int voxels_N, bool filterDetector, ObbData detector_volume, bool leadShield_present, ObbData leadShield_volume )
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


    // Read current particle's energy
    f32 E = particles->E[ id ]; // / MeV;
    f32 E_sq = E * E;

    // Contsant index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / spacing;
    ivoxsize.y = 1.0 / spacing;
    ivoxsize.z = 1.0 / spacing;

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

	// Check if particle is touching the detector
	if( filterDetector )
	{
        	f32 dist = hit_ray_OBB( pos, dir, detector_volume );

	        if( dist < spacing )
        	{
        	    particle_in_grid = false;
		}
        }

	// Check if particle is touching the lead shield
	if( leadShield_present )
	{
        	f32 dist = hit_ray_OBB( pos, dir, leadShield_volume );

	        if( dist < spacing )
        	{
        	    particle_in_grid = false;
		}
        }

    }

}


// Digitizer record and process data into the detector. For example in CT imaging the digitizer will compute
// the number of particle per pixel.
__host__ __device__ void scatterSimulator_digitizer( ParticlesData *particles, ui32 id )
{
    // If freeze or dead, quit
    if( particles->status[ id ] == PARTICLE_FREEZE || particles->status[ id ] == PARTICLE_DEAD )
    {
        return;
    }

}

// Kernel that launch the function track_to_in on GPU
__global__ void kernel_scatterSimulator_track_to_in( ParticlesData *particles, float* gridDimensions, int spacing, int vox_x, int vox_y, float* energiesPerVoxel, float* energiesPerVoxel_sq, float* hitsPerVoxel, int voxels_N, bool filterDetector, ObbData detector_volume, bool leadShield_present, ObbData leadShield_volume )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles->size) return;

    scatterSimulator_track_to_in( particles, id, gridDimensions, spacing, vox_x, vox_y, energiesPerVoxel, energiesPerVoxel_sq, hitsPerVoxel, voxels_N, filterDetector, detector_volume, leadShield_present, leadShield_volume );
}


// Kernel that launch digitizer on GPU
__global__ void kernel_scatterSimulator_digitizer( ParticlesData *particles )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles->size) return;

    scatterSimulator_digitizer( particles, id );
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

ScatterSimulator::ScatterSimulator(bool save_all_data) : GGEMSDetector(),
                                      m_width( 0.0f ),
                                      m_height( 0.0f ),
                                      m_depth( 0.0f ),
				      m_verbose_on( save_all_data )
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

    m_proj_axis.m00 = 1.0;
    m_proj_axis.m01 = 0.0;
    m_proj_axis.m02 = 0.0;
    m_proj_axis.m10 = 0.0;
    m_proj_axis.m11 = 1.0;
    m_proj_axis.m12 = 0.0;
    m_proj_axis.m20 = 0.0;
    m_proj_axis.m21 = 0.0;
    m_proj_axis.m22 = 1.0;

    m_detector_built = false;
    m_filter_detector = false;
    m_save_all_data = false;

    m_leadShield_present = false;
    m_leadShield_built = false;
    m_leadShield_rotation = make_f32xyz( 0, 0, 0 );
}

ScatterSimulator::~ScatterSimulator()
{
    delete []m_energies_per_Voxel_h;
    delete []m_energies_per_Voxel_squared_h;
    delete []m_hits_per_Voxel_h;
    delete []m_uncertainty_per_Voxel;
    delete []m_voxelizedGriddims_h;
    delete []m_energies_per_Voxel_smoothed;
}

//============ Setting functions ==========================

void ScatterSimulator::set_grid_dimension( f32 width, f32 height, f32 depth )
{
    m_width = width;
    m_height = height;
    m_depth = depth;
}

void ScatterSimulator::set_detector_position(f32 center_x, f32 center_y, f32 center_z)
{
    m_detectorPos = make_float3(center_x, center_y, center_z);
}

void ScatterSimulator::set_detector_rotation( f32 rx, f32 ry, f32 rz )
{
    m_detector_rotation = make_f32xyz( rx, ry, rz );
}

void ScatterSimulator::set_detector_size( f32 width, f32 height, f32 depth )
{
    m_detector_size = make_f32xyz( width, height, depth );
    this->build_detector_volume();
}

void ScatterSimulator::build_detector_volume()
{
    m_detector_volume.xmin = -m_detector_size.x * 0.5;
    m_detector_volume.xmax = m_detector_size.x * 0.5;
    m_detector_volume.ymin = -m_detector_size.y * 0.5;
    m_detector_volume.ymax = m_detector_size.y * 0.5;
    m_detector_volume.zmin = -m_detector_size.z * 0.5;
    m_detector_volume.zmax = m_detector_size.z * 0.5;

    TransformCalculator *trans = new TransformCalculator;
    trans->set_translation( m_detectorPos.x, m_detectorPos.y, m_detectorPos.z );
    trans->set_rotation( m_detector_rotation );
    trans->set_axis_transformation( m_proj_axis );
    m_detector_volume.transformation = trans->get_transformation_matrix();
    delete trans;

    m_detector_built = true;

    if( m_verbose_on )
    {    
    	GGcout << "Detector's position: " << m_detectorPos.x << "\t" << m_detectorPos.y << "\t" << m_detectorPos.z << GGendl;
    	GGcout << "Detector's vol: " << m_detector_volume.xmin << "\t" << m_detector_volume.xmax << "\t" << m_detector_volume.ymin << "\t" << m_detector_volume.ymax << "\t" << m_detector_volume.zmin << "\t" << m_detector_volume.zmax << GGendl;
    	GGcout << m_detector_volume.transformation << GGendl;
    }
}

void ScatterSimulator::set_leadShield_position(f32 center_x, f32 center_y, f32 center_z)
{
    m_leadShieldPos = make_float3(center_x, center_y, center_z);
    m_leadShield_present = true;
}

void ScatterSimulator::set_leadShield_rotation( f32 rx, f32 ry, f32 rz )
{
    m_leadShield_rotation.x = rx;
    m_leadShield_rotation.y = ry;
    m_leadShield_rotation.z = rz;
}

void ScatterSimulator::set_leadShield_size( f32 width, f32 height, f32 depth )
{
    m_leadShield_size = make_f32xyz( width, height, depth );
    this->build_leadShield_volume();
}

void ScatterSimulator::build_leadShield_volume()
{
    m_leadShield_volume.xmin = -m_leadShield_size.x * 0.5;
    m_leadShield_volume.xmax = m_leadShield_size.x * 0.5;
    m_leadShield_volume.ymin = -m_leadShield_size.y * 0.5;
    m_leadShield_volume.ymax = m_leadShield_size.y * 0.5;
    m_leadShield_volume.zmin = -m_leadShield_size.z * 0.5;
    m_leadShield_volume.zmax = m_leadShield_size.z * 0.5;

    TransformCalculator *trans = new TransformCalculator;
    trans->set_translation( m_leadShieldPos.x, m_leadShieldPos.y, m_leadShieldPos.z );
    trans->set_rotation( m_leadShield_rotation );
    trans->set_axis_transformation( m_proj_axis );
    m_leadShield_volume.transformation = trans->get_transformation_matrix();
    delete trans;

    m_leadShield_built = true;

    if( m_verbose_on )
    {    
    	GGcout << "Lead shield's center position: " << m_leadShieldPos.x << "\t" << m_leadShieldPos.y << "\t" << m_leadShieldPos.z << GGendl;
    	GGcout << "Lead shield's vol: " << m_leadShield_volume.xmin << "\t" << m_leadShield_volume.xmax << "\t" << m_leadShield_volume.ymin << "\t" << m_leadShield_volume.ymax << "\t" << m_leadShield_volume.zmin << "\t" << m_leadShield_volume.zmax << GGendl;
    	GGcout << m_leadShield_volume.transformation << GGendl;
    }
}

void ScatterSimulator::set_grid_center( f32 x, f32 y, f32 z )
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

    if( m_verbose_on )
    { 
	GGcout<<"*******Voxelized Grid's Position set " << GGendl;
	GGcout<<"*******xmin " << m_voxelizedGriddims_h[0] << " xmax " << m_voxelizedGriddims_h[1] << GGendl;
	GGcout<<"*******ymin " << m_voxelizedGriddims_h[2] << " ymax " << m_voxelizedGriddims_h[3] << GGendl;
	GGcout<<"*******zmin " << m_voxelizedGriddims_h[4] << " zmax " << m_voxelizedGriddims_h[5] << GGendl;
    }
}

void ScatterSimulator::buildVoxelGrid()
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
    m_energies_per_Voxel_smoothed = new float[ m_nb_of_Voxels ];

    GGcout<<"*******Voxelized Scatter Detector was BUILT " << GGendl;
}

//============ Some functions ============================

void ScatterSimulator::setDataOutputPath( std::string filename )
{    
    m_dataOutputPath = filename;
}

// Export data
void ScatterSimulator::save_data( int smoothFactor )
{
    //std::string filename_stats = m_dataOutputPath + "_ScatterE.txt";

    // Copy data from GPU to CPU
    HANDLE_ERROR( cudaMemcpy(m_energies_per_Voxel_h, m_energies_per_Voxel_d, m_nb_of_Voxels * sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(m_energies_per_Voxel_squared_h, m_energies_per_Voxel_squared_d, m_nb_of_Voxels * sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(m_hits_per_Voxel_h, m_hits_per_Voxel_d, m_nb_of_Voxels * sizeof(float), cudaMemcpyDeviceToHost) );

    // Save results
    //write_voxGrid_data( filename_stats, m_energies_per_Voxel_h, m_energies_per_Voxel_squared_h, m_hits_per_Voxel_h, m_nb_of_Voxels );

    // Create IO object
    ImageIO *im_io = new ImageIO;
    std::string filename_E = m_dataOutputPath + "_Energy.mhd";

    f32xyz voxSpacing = make_float3( m_voxelSpacing, m_voxelSpacing, m_voxelSpacing );
    f32xyz offset = make_float3( m_width * (-0.5), m_height * (-0.5), m_depth * (-0.5) );  
    ui32xyz num_Vox = make_ui32xyz( m_voxels_x, m_voxels_y, m_voxels_z );

    float *eperVox = new float[ m_nb_of_Voxels ];
    float *eperVoxRaw = new float[ m_nb_of_Voxels ];
    float *doseperVox = new float[ m_nb_of_Voxels ];
    float *eSqperVox = new float[ m_nb_of_Voxels ];
    ui32 *hitsperVox = new ui32[ m_nb_of_Voxels ];

    float density_air = 0.00129;
    float volume = m_voxelSpacing/10 * m_voxelSpacing/10 * m_voxelSpacing/10; //! volume in cm^3
    float mass = (density_air * volume) / 1000; //! mass in kg

    // Smoothing of recorded energy values
    if( smoothFactor != 0 || smoothFactor != 1 )
	this->smoothRecordedValues( smoothFactor );

    for( ui32 i = 0; i < m_nb_of_Voxels; ++i )
    {
	float E = m_energies_per_Voxel_h[ i ];
	eperVox[ i ] = E;	

	if( m_save_all_data )
	{
		//! Convert Energy value to dose to Air
		float energy_J = E * 1.60218e-19; //! Convert Kev to J     	
    		float dose = energy_J / mass; //! Dose to Air in Gy
    		dose*=1000; //! to mGy        
		doseperVox[ i ] = dose;		
		eSqperVox[ i ] = m_energies_per_Voxel_squared_h[ i ];
		hitsperVox[ i ] = m_hits_per_Voxel_h[ i ];
		eperVoxRaw[ i ] = m_energies_per_Voxel_smoothed[ i ];
	}
	
    }
    // Save Energy
    im_io->write_3D( filename_E, eperVox, num_Vox, offset, voxSpacing );
    
    if( m_save_all_data )
    {
    	std::string filename_D = m_dataOutputPath + "_Dose.mhd";
    	std::string filename_Esq = m_dataOutputPath + "_Esq.mhd";
    	std::string filename_H = m_dataOutputPath + "_Hits.mhd";
    	std::string filename_Raw = m_dataOutputPath + "_Eraw.mhd";
	im_io->write_3D( filename_D, doseperVox, num_Vox, offset, voxSpacing );
	im_io->write_3D( filename_Esq, eSqperVox, num_Vox, offset, voxSpacing );
	im_io->write_3D( filename_H, hitsperVox, num_Vox, offset, voxSpacing );
	im_io->write_3D( filename_Raw, eperVoxRaw, num_Vox, offset, voxSpacing );
	GGcout << "**/*/* Dose per voxel saved in "<< filename_D << GGendl;
	GGcout << "**/*/* Energy squared per voxel saved in "<< filename_Esq << GGendl;
	GGcout << "**/*/* Hits per voxel saved in "<< filename_H << GGendl;
	GGcout << "**/*/* Raw Energy per voxel saved in "<< filename_Raw << GGendl;
    }

    delete[] eperVox;
    delete[] doseperVox;
    delete[] eperVoxRaw;
    delete[] eSqperVox;
    delete[] hitsperVox;

    //GGcout << "**/*/* Voxelized grid results saved in "<< filename_stats << GGendl;
    GGcout << "**/*/* Energy per voxel saved in "<< filename_E << GGendl;

}

void ScatterSimulator::smoothRecordedValues( int smoothFactor )
{
    // Compute grids min, max and mean values
    float Emin = FLT_MAX;
    float Emax = 0.0;
    float Emean = 0.0;
    float sumE = 0.0;
    ui32 numElements = 0;
    for( ui32 i = 0; i < m_nb_of_Voxels; ++i )
    {
        float data_i = m_energies_per_Voxel_h[ i ];
        if(data_i != 0)
        {
            sumE += data_i;
            if(data_i > Emax)
                Emax = data_i;
            if(data_i < Emin)
                Emin = data_i;
            numElements++;
        }
    }
    if(numElements > 0)
        Emean = sumE / numElements;

    // Smooth factors
    //! High threshold
    float th_high = Emean * smoothFactor;

    //! Low threshold
    float th_low = Emean / smoothFactor;

    for( ui32 i = 0; i < m_nb_of_Voxels; ++i )
    {
	m_energies_per_Voxel_smoothed[ i ] =  m_energies_per_Voxel_h[ i ];       
	if(m_energies_per_Voxel_h[ i ] >= th_high)
            m_energies_per_Voxel_h[ i ] = th_high;

        if(m_energies_per_Voxel_h[ i ] <= th_low)
            m_energies_per_Voxel_h[ i ] = th_low;
    }
}

//============ Mandatory functions =======================

// Check if everything is ok to initialize this detector
bool ScatterSimulator::m_check_mandatory()
{
    if ( m_width == 0.0 || m_height == 0.0 || m_depth == 0.0 || m_voxelSpacing == 0.0 || m_detector_built == false ) return false;
    else return true;
}

// This function is mandatory and called by GGEMS to initialize and load all
// necessary data on the graphic card
void ScatterSimulator::initialize( GlobalSimulationParametersData *params )
{
    // Check the parameters
    if( !m_check_mandatory() )
    {
        GGcerr << "Voxelized scatter detector was not set properly!" << GGendl;
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

    GGcout << "Voxelized scatter detector correctly initialized" << GGendl;
}

// Mandatory function, that handle track_to_in
void ScatterSimulator::track_to_in( ParticlesData *d_particles )
{

    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 )
             / mh_params->gpu_block_size;


    // Execution og GPU Kernel function
    kernel_scatterSimulator_track_to_in<<<grid, threads>>>( d_particles, m_voxelizedGriddims_d, m_voxelSpacing, m_voxels_y, m_voxels_z, m_energies_per_Voxel_d, m_energies_per_Voxel_squared_d, m_hits_per_Voxel_d, m_nb_of_Voxels, m_filter_detector, m_detector_volume, m_leadShield_present, m_leadShield_volume );
    cuda_error_check("Error ", " Kernel_template_detector (track to in)");
    cudaThreadSynchronize();

}

// If navigation within the detector is required te track_to_out function should be
// equivalent to the track_to_in function. Here there is no navigation. However, this function
// is mandatory, and must be defined
void ScatterSimulator::track_to_out( ParticlesData *d_particles ) {}

// Same mandatory function to drive the digitizer function between CPU and GPU
void ScatterSimulator::digitizer( ParticlesData *d_particles )
{
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 )
             / mh_params->gpu_block_size;

    kernel_scatterSimulator_digitizer<<<grid, threads>>>( d_particles );
    cuda_error_check("Error ", " Kernel_template_detector (digitizer)");
    cudaThreadSynchronize();
}



#endif



