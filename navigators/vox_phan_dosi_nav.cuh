// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_dosi.cuh
 * \brief
 * \author Y. Lemaréchal
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef VOX_PHAN_DOSI_NAV_CUH
#define VOX_PHAN_DOSI_NAV_CUH

#include "global.cuh"
#include "ggems_phantom.cuh"
#include "voxelized.cuh"
#include "raytracing.cuh"
#include "vector.cuh"
#include "materials.cuh"
#include "photon.cuh"
#include "photon_navigator.cuh"
#include "image_reader.cuh"
#include "dose_calculator.cuh"
#include "electron.cuh"
#include "cross_sections.cuh"
#include "electron_navigator.cuh"
#include "transport_navigator.cuh"

// VoxPhanDosiNav -> VPDN
namespace VPDN
{
__host__ __device__ void track_to_in ( ParticlesData &particles, f32 xmin, f32 xmax,
                                       f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                                       ui32 id );

__host__ __device__ void track_electron_to_out ( ParticlesData &particles,
        VoxVolumeData vol,
        MaterialsTable materials,        
        ElectronsCrossSectionTable electron_CS_table,
        GlobalSimulationParametersData parameters,
        DoseData &dosi,
        f32 &randomnumbereIoni,
        f32 &randomnumbereBrem,
        f32 freeLength,
        ui32 part_id );

__host__ __device__ void track_photon_to_out ( ParticlesData &particles,
        VoxVolumeData vol,
        MaterialsTable materials,
        PhotonCrossSectionTable photon_CS_table,        
        GlobalSimulationParametersData parameters,
        DoseData dosi,
        ui32 part_id );

__global__ void kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
        f32 ymin, f32 ymax, f32 zmin, f32 zmax );

__global__ void kernel_device_track_to_out ( ParticlesData particles,
        VoxVolumeData vol,
        MaterialsTable materials,
        PhotonCrossSectionTable photon_CS_table,
        ElectronsCrossSectionTable electron_CS_table,
        GlobalSimulationParametersData parameters,
        DoseData dosi );

void kernel_host_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                               f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 part_id );

void kernel_host_track_to_out ( ParticlesData particles,
                                VoxVolumeData vol,
                                MaterialsTable materials,
                                PhotonCrossSectionTable photon_CS_table,
                                ElectronsCrossSectionTable electron_CS_table,
                                GlobalSimulationParametersData parameters,
                                DoseData dosi,
                                ui32 id );
}

class VoxPhanDosiNav : public GGEMSPhantom
{
public:
    VoxPhanDosiNav() {}
    ~VoxPhanDosiNav() {}

    // Init
    void initialize ( GlobalSimulationParameters params );
    // Tracking from outside to the phantom broder
    void track_to_in ( Particles particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out ( Particles particles );

    void load_phantom_from_mhd ( std::string filename, std::string range_mat_name );

    void calculate_dose_to_phantom();
    void calculate_dose_to_water();
    
    void write ( std::string filename = "dosimetry.mhd" );
    void set_elements( std::string filename );
    void set_materials( std::string filename );
private:

    VoxelizedPhantom m_phantom;
    Materials m_materials;
    CrossSections m_cross_sections;
    DoseCalculator m_dose_calculator;

    bool m_check_mandatory();

    GlobalSimulationParameters m_params;

    std::string m_elements_filename;
    std::string m_materials_filename;

//
};

#endif
