// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_gtrack_nav.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 08/11/2016
 *
 *
 *
 */

#ifndef VOX_PHAN_GTRACK_NAV_CUH
#define VOX_PHAN_GTRACK_NAV_CUH

#include "ggems.cuh"
#include "global.cuh"
#include "ggems_phantom.cuh"
#include "voxelized.cuh"
#include "raytracing.cuh"
#include "vector.cuh"
#include "materials.cuh"
#include "photon.cuh"
#include "photon_navigator.cuh"
#include "image_io.cuh"
#include "dose_calculator.cuh"
//#include "electron.cuh"
//#include "cross_sections.cuh"
//#include "electron_navigator.cuh"
#include "transport_navigator.cuh"
//#include "mu_data.cuh"

// GTrackModelData
struct GTrackUncorrelatedModelData {
    f32 *bin_energy;
    f32 *bin_step;
    f32 *bin_edep;
    f32 *bin_scatter;

    f32 *cdf_step;
    f32 *cdf_edep;
    f32 *cdf_scatter;

    ui16 *lcdf_step;
    ui16 *lcdf_edep;
    ui16 *lcdf_scatter;

    f32 di_energy;
    f32 di_lut;
    f32 min_E;

    ui32 nb_bins;
    ui32 nb_energy_bins;
    ui32 nb_lut_bins;
};

// VoxPhanGTrackNav -> VPGTN
namespace VPGTN
{

__host__ __device__ void track_to_out_uncorrelated_model( ParticlesData particles,
                                                          VoxVolumeData<ui16> vol,
                                                          GTrackUncorrelatedModelData model,
                                                          GlobalSimulationParametersData parameters, ui32 part_id);

__global__ void kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                            f32 ymin, f32 ymax, f32 zmin, f32 zmax , f32 tolerance);

__global__ void kernel_device_track_to_out_uncorrelated_model ( ParticlesData particles,
                                                                 VoxVolumeData<ui16> vol,
                                                                 GTrackUncorrelatedModelData model,
                                                                 GlobalSimulationParametersData parameters );


//void kernel_host_track_to_in( ParticlesData particles, f32 xmin, f32 xmax,
//                               f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 part_id );

//void kernel_host_track_to_out( ParticlesData particles,
//                               VoxVolumeData<ui16> vol,
//                               GTrackModelData model,
//                               GlobalSimulationParametersData parameters );

}

class VoxPhanGTrackNav : public GGEMSPhantom
{
public:
    VoxPhanGTrackNav();
    ~VoxPhanGTrackNav() {}

    // Init
    void initialize( GlobalSimulationParameters params );
    // Tracking from outside to the phantom border
    void track_to_in( Particles particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out( Particles particles );

    void load_phantom_from_mhd( std::string filename, std::string range_mat_name );

    void load_gtrack_uncorrelated_model( std::string filename );

    //void calculate_dose_to_phantom();
    //void calculate_dose_to_water();
    
    //void write( std::string filename = "dosimetry.mhd" );
    void set_materials( std::string filename );       

    //VoxVolumeData<f32> * get_dose_map();

    AabbData get_bounding_box();

private:

    VoxelizedPhantom m_phantom;
    //Materials m_materials;
    //CrossSections m_cross_sections;
    //DoseCalculator m_dose_calculator;
    GTrackUncorrelatedModelData m_gtrack_uncorrelated_model;

    bool m_check_mandatory();

    // Get the memory usage
    ui64 m_get_memory_usage();

    f32 m_dosel_size_x, m_dosel_size_y, m_dosel_size_z;
    f32 m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax;

    GlobalSimulationParameters m_params;

    std::string m_materials_filename;

//
};

#endif
