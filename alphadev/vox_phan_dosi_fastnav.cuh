// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_dosi.cuh
 * \brief
 * \author J. Bert
 * \version 0.1
 * \date January 4th 2017
 *
 *
 *
 */

#ifndef VOX_PHAN_DOSI_FASTNAV_CUH
#define VOX_PHAN_DOSI_FASTNAV_CUH

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

// Electrons buffer in SoA ( Acces to level : Part_ID * size + hierarchy level )
struct ElectronsBuffer
{
    // properties
    f32* E;    // N
    f32* dx;
    f32* dy;
    f32* dz;
    f32* px;
    f32* py;
    f32* pz;

    // Level free index
    ui8 *level_free;        // P

    // warning vector
    ui8 *warning;           // P

    // size
    ui32 size_of_particles; // P
    ui32 size_of_buffer;    // N
    ui8 level_max;
}; //

// VoxPhanDosiFastNav -> VPDFN
namespace VPDFN
{

__host__ __device__ void electrons_track_to_out( ParticlesData particles,
                                                 VoxVolumeData<ui16> vol,
                                                 MaterialsTable materials,
                                                 ElectronsCrossSectionTable electron_CS_table,
                                                 GlobalSimulationParametersData parameters,
                                                 DoseData dosi,
                                                 f32 &randomnumbereIoni,
                                                 f32 &randomnumbereBrem,
                                                 f32 &freeLength,
                                                 ui32 part_id );

__host__ __device__ void photons_track_to_out( ParticlesData particles,
                                               ElectronsBuffer e_buffer,
                                               VoxVolumeData<ui16> vol,
                                               MaterialsTable materials,
                                               PhotonCrossSectionTable photon_CS_table,
                                               GlobalSimulationParametersData parameters,
                                               DoseData dosi,
                                               ui32 part_id );

__global__ void kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                           f32 ymin, f32 ymax, f32 zmin, f32 zmax , f32 tolerance);

__global__ void kernel_photons_track_to_out( ParticlesData particles,
                                             ElectronsBuffer e_buffer,
                                             VoxVolumeData<ui16> vol,
                                             MaterialsTable materials,
                                             PhotonCrossSectionTable photon_CS_table,
                                             GlobalSimulationParametersData parameters,
                                             DoseData dosi );

__global__ void kernel_electrons_track_to_out( ParticlesData particles, ElectronsBuffer e_buffer,
                                               VoxVolumeData<ui16> vol,
                                               MaterialsTable materials,
                                               ElectronsCrossSectionTable electron_CS_table,
                                               GlobalSimulationParametersData parameters,
                                               DoseData dosi );

void kernel_host_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                               f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 part_id );

void kernel_host_track_to_out (ParticlesData particles,
                               VoxVolumeData<ui16> vol,
                               MaterialsTable materials,
                               PhotonCrossSectionTable photon_CS_table,
                               ElectronsCrossSectionTable electron_CS_table,
                               GlobalSimulationParametersData parameters,
                               DoseData dosi,
                               ui32 id );
}

class VoxPhanDosiFastNav : public GGEMSPhantom
{
public:
    VoxPhanDosiFastNav();
    ~VoxPhanDosiFastNav();

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
    void set_materials( std::string filename );
    void set_dosel_size( f32 sizex, f32 sizey, f32 sizez );
    void set_dose_min_density( f32 min );
    void set_volume_of_interest( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax );

    void set_number_of_stored_electrons_per_photons( ui8 nb );

    void export_density_map( std::string filename );
    void export_materials_map( std::string filename );
    void export_electrons_buffer( std::string filename );

    AabbData get_bounding_box();

private:

    VoxelizedPhantom m_phantom;
    Materials m_materials;
    CrossSections m_cross_sections;
    DoseCalculator m_dose_calculator;
    ElectronsBuffer m_electrons_buffer;

    bool m_check_mandatory();

    // Get the memeory usage
    ui64 m_get_memory_usage();
    void m_init_electrons_buffer();

    f32 m_dosel_size_x, m_dosel_size_y, m_dosel_size_z;
    f32 m_dose_min_density;
    f32 m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax;

    GlobalSimulationParameters m_params;

    std::string m_materials_filename;

//
};

#endif
