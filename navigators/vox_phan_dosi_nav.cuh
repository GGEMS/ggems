// GGEMS Copyright (C) 2017

/*!
 * \file vox_phan_dosi.cuh
 * \brief
 * \author Y. Lemaréchal
 * \version 0.2
 * \date 18 novembre 2015
 *
 * v0.2: JB - Change all structs and remove CPU exec
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
#include "image_io.cuh"
#include "dose_calculator.cuh"
#include "electron.cuh"
#include "cross_sections.cuh"
#include "electron_navigator.cuh"
#include "transport_navigator.cuh"
#include "primitives.cuh"

// VoxPhanDosiNav -> VPDN
namespace VPDN
{

__host__ __device__ void track_electron_to_out( ParticlesData *particles,
                                                ParticlesData *buffer,
                                                const VoxVolumeData<ui16> *vol,
                                                const MaterialsData *materials,
                                                const ElectronsCrossSectionData *electron_CS_table,
                                                const GlobalSimulationParametersData *parameters,
                                                DoseData *dosi,
                                                f32 &randomnumbereIoni,
                                                f32 &randomnumbereBrem,
                                                f32 &freeLength,
                                                ui32 part_id );

__host__ __device__ void track_photon_to_out( ParticlesData *particles, ParticlesData *buffer,
                                              const VoxVolumeData<ui16> *vol,
                                              const MaterialsData *materials,
                                              const PhotonCrossSectionData *photon_CS_table,
                                              const GlobalSimulationParametersData *parameters,
                                              DoseData *dosi,
                                              ui32 part_id );

__global__ void kernel_device_track_to_in( ParticlesData *particles, f32 xmin, f32 xmax,
                                           f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance );

__global__ void kernel_device_track_to_out( ParticlesData *particles,
                                            ParticlesData *buffer,
                                            const VoxVolumeData<ui16> *vol,
                                            const MaterialsData *materials,
                                            const PhotonCrossSectionData *photon_CS_table,
                                            const ElectronsCrossSectionData *electron_CS_table,
                                            const GlobalSimulationParametersData *parameters,
                                            DoseData *dosi );

}

class VoxPhanDosiNav : public GGEMSPhantom
{
public:
    VoxPhanDosiNav();
    ~VoxPhanDosiNav() {}

    // Init
    void initialize( GlobalSimulationParametersData *h_params, GlobalSimulationParametersData *d_params );
    // Tracking from outside to the phantom broder
    void track_to_in( ParticlesData *d_particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out( ParticlesData *d_particles );

    void load_phantom_from_mhd( std::string filename, std::string range_mat_name );

    void calculate_dose_to_medium();
    void calculate_dose_to_water();
    
    void write( std::string filename = "dosimetry.mhd", std::string options = "all" );
    void set_materials( std::string filename );
    void set_dosel_size( f32 sizex, f32 sizey, f32 sizez );
    void set_dose_min_density( f32 min );
    void set_volume_of_interest( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax );

    void export_density_map( std::string filename );
    void export_materials_map( std::string filename );

    AabbData get_bounding_box();

    void update_clear_deposition();

private:

    VoxelizedPhantom m_phantom;
    Materials m_materials;
    CrossSections m_cross_sections;
    DoseCalculator m_dose_calculator;
    ParticleManager m_particles_buffer;

    bool m_check_mandatory();

    // Get the memeory usage
    ui64 m_get_memory_usage();

    f32 m_dosel_size_x, m_dosel_size_y, m_dosel_size_z;
    f32 m_dose_min_density;
    f32 m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax;

    GlobalSimulationParametersData *mh_params;
    GlobalSimulationParametersData *md_params;

    std::string m_materials_filename;

//
};

#endif
