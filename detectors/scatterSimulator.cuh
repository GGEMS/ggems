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

class ScatterSimulator : public GGEMSDetector
{
    public:
        ScatterSimulator( bool verboseON = false );
        ~ScatterSimulator();

        // Setting
        void set_grid_dimension( f32 width, f32 height, f32 depth );
	void set_grid_center(f32 center_x, f32 center_y, f32 center_z);
	void set_voxel_spacing( int s ){ m_voxelSpacing = s; }

	void set_detector_position(f32 center_x, f32 center_y, f32 center_z);
	void set_detector_rotation( f32 rx, f32 ry, f32 rz );
	void set_detector_size(f32 width, f32 height, f32 depth);
	void set_detector_filtering( ){ m_filter_detector = true; }

	void set_leadShield_position(f32 center_x, f32 center_y, f32 center_z);
	void set_leadShield_rotation( f32 rx, f32 ry, f32 rz );
	void set_leadShield_size(f32 width, f32 height, f32 depth);

        // Mandatory functions from abstract class GGEMSDetector
        void initialize( GlobalSimulationParametersData *h_params );   // Initialisation
        void track_to_in( ParticlesData *d_particles );                // Navigation until the detector
        void track_to_out( ParticlesData *d_particles );               // Navigation within the detector
        void digitizer( ParticlesData *d_particles );                  // Hits processing into data (histo, image, etc.)

        // Save data
        void save_data( int smoothFactor = 0 );
	void setDataOutputPath( std::string filename );
	void setSaveAllData( bool t ){ m_save_all_data = t; }

    private:
        bool m_check_mandatory();
	bool m_save_all_data;
	bool m_verbose_on;

	// Voxelized grid properties
        f32 m_width, m_height, m_depth;
	float* m_voxelizedGriddims_h; // [xmin, xmax, ymin, ymax, zmin, zmax]
	float* m_voxelizedGriddims_d;
	f32xyz m_grid_center;
	int m_voxels_x, m_voxels_y, m_voxels_z, m_voxelSpacing, m_nb_of_Voxels;
	void buildVoxelGrid();

	// Detector parameters
	bool m_filter_detector;
	f32xyz m_detectorPos;
	ObbData m_detector_volume;
	f32xyz m_detector_size;
	f32xyz m_detector_rotation;
	f32matrix44 m_detector_transform;
	f32matrix33 m_proj_axis;
	void build_detector_volume();
	bool m_detector_built;

	// Lead shield parameters
	void build_leadShield_volume();
	f32xyz m_leadShield_size;
	f32xyz m_leadShieldPos;
    	f32xyz m_leadShield_rotation;
	bool m_leadShield_present;
	bool m_leadShield_built;
	ObbData m_leadShield_volume;

	// Simulation parameters
        GlobalSimulationParametersData *mh_params;
	ui64 m_nb_of_particles;
	ui64 m_batch_size;

	// Output results
	std::string m_dataOutputPath;
	float* m_energies_per_Voxel_h;
	float* m_energies_per_Voxel_d;
	float* m_energies_per_Voxel_squared_h;
	float* m_energies_per_Voxel_squared_d;
	float* m_hits_per_Voxel_h;
	float* m_hits_per_Voxel_d;
	float* m_uncertainty_per_Voxel;
	float* m_energies_per_Voxel_smoothed;

	void smoothRecordedValues( int smoothFactor );


};

#endif

