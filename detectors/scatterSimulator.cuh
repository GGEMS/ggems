/*
  (C) Copyright University of Strasbourg, All Rights Reserved.
*/


#ifndef SCATTERSIMULATOR_CUH
#define SCATTERSIMULATOR_CUH

#include "global.cuh"
#include "raytracing.cuh"
#include "particles.cuh"
#include "primitives.cuh"
#include "fun.cuh"
#include "image_io.cuh"
#include "ggems_detector.cuh"
#include "transport_navigator.cuh"

class GGEMSDetector;

class VoxGridDetector : public GGEMSDetector
{
    public:
        VoxGridDetector();
        ~VoxGridDetector();

        // Setting
        void set_dimension( f32 width, f32 height, f32 depth );
	void set_grid_center(f32 center_x, f32 center_y, f32 center_z);
	void set_source_position(f32 center_x, f32 center_y, f32 center_z);
	void set_source_aperture(f32 ap){m_source_ap = ap;}// * ( gpu_pi / 180.0f );}
	void set_source_detector_dist(f32 d){m_source_h = d;}
	void set_voxel_spacing( int s ){ m_voxelSpacing = s; }

        // Mandatory functions from abstract class GGEMSDetector
        void initialize( GlobalSimulationParametersData *h_params );   // Initialisation
        void track_to_in( ParticlesData *d_particles );                // Navigation until the detector
        void track_to_out( ParticlesData *d_particles );               // Navigation within the detector
        void digitizer( ParticlesData *d_particles );                  // Hits processing into data (histo, image, etc.)

        // Save data
        void save_data(  );
	void setDataOutputPath( std::string filename );

    private:
        bool m_check_mandatory();

	// Voxelized grid properties
        f32 m_width, m_height, m_depth;
	float* m_voxelizedGriddims_h; // [xmin, xmax, ymin, ymax, zmin, zmax]
	float* m_voxelizedGriddims_d;
	f32xyz m_grid_center;
	int m_voxels_x, m_voxels_y, m_voxels_z, m_voxelSpacing, m_nb_of_Voxels;
	void buildVoxelGrid();

	// Simulation parameters
        GlobalSimulationParametersData *mh_params;
	ui64 m_nb_of_particles;
	ui64 m_batch_size;
	f32xyz m_source_p;
	f32 m_source_h;
	f32 m_source_ap;
	f32xyz m_detectorPos;

	// Output results
	std::string m_dataOutputPath;
	float* m_energies_per_Voxel_h;
	float* m_energies_per_Voxel_d;
	float* m_energies_per_Voxel_squared_h;
	float* m_energies_per_Voxel_squared_d;
	float* m_hits_per_Voxel_h;
	float* m_hits_per_Voxel_d;
	float* m_uncertainty_per_Voxel;

	float3* m_particlesPos_h;
	float3* m_particlesPos_d;

};

#endif
