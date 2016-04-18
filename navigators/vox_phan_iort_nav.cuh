// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_iort_nav.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 23/03/2016
 *
 *
 *
 */

#ifndef VOX_PHAN_IORT_NAV_CUH
#define VOX_PHAN_IORT_NAV_CUH

#include "ggems.cuh"
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
#include "mu_data.cuh"

//#define SKIP_VOXEL

// For variance reduction (use in IORT for instance)
#define analog 0
#define TLE    1
#define seTLE  2

// Mu and Mu_en table used by TLE
struct Mu_MuEn_Table{
    ui32 nb_mat;      // k
    ui32 nb_bins;     // n

    f32 E_min;
    f32 E_max;

    f32* E_bins;      // n
    f32* mu;          // n*k
    f32* mu_en;       // n*k

    ui8 flag;        // type of TLE? 0- Not used, 1- TLE, 2- seTLE
};

// History map used by seTLE
struct HistoryMap {
    ui32 *interaction;
    f32 *energy;
};

// COO compression history map used by seTLE
struct COOHistoryMap {
    ui16 *x;
    ui16 *y;
    ui16 *z;
    f32 *energy;
    ui32 *interaction;

    ui32 nb_data;
};

/*
// Cylinder structure for brachy seeds
struct CylinderTransform
{
    f32 *tx, *ty, *tz;
    f32 *rx, *ry, *rz;
    f32 *sx, *sy, *sz;

    f32 *cdf;

    ui32 nb_sources;
};
*/

// VoxPhanIORTNav -> VPIN
namespace VPIORTN
{

__host__ __device__ void track_to_out(ParticlesData &particles,
                                      VoxVolumeData vol, MaterialsTable materials,
                                      PhotonCrossSectionTable photon_CS_table,
                                      GlobalSimulationParametersData parameters, DoseData dosi, Mu_MuEn_Table mu_table,
                                      HistoryMap hist_map, ui32 part_id);

__host__ __device__ void track_seTLE(ParticlesData &particles, VoxVolumeData vol, COOHistoryMap coo_hist_map,
                                      DoseData dose, Mu_MuEn_Table mu_table, ui32 nb_of_rays, f32 edep_th, ui32 id );

__global__ void kernel_device_track_to_in (ParticlesData particles, f32 xmin, f32 xmax,
                                            f32 ymin, f32 ymax, f32 zmin, f32 zmax , f32 tolerance);

__global__ void kernel_device_track_to_out (ParticlesData particles,
                                             VoxVolumeData vol,
                                             MaterialsTable materials,
                                             PhotonCrossSectionTable photon_CS_table,
                                             GlobalSimulationParametersData parameters,
                                             DoseData dosi , Mu_MuEn_Table mu_table, HistoryMap hist_map);

__global__ void kernel_device_seTLE( ParticlesData particles, VoxVolumeData vol,
                                     COOHistoryMap coo_hist_map,
                                     DoseData dosi,
                                     Mu_MuEn_Table mu_table, ui32 nb_of_rays, f32 edep_th );

void kernel_host_track_to_in (ParticlesData particles, f32 xmin, f32 xmax,
                               f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 tolerance, ui32 part_id );

void kernel_host_track_to_out (ParticlesData particles,
                                VoxVolumeData vol,
                                MaterialsTable materials,
                                PhotonCrossSectionTable photon_CS_table,
                                GlobalSimulationParametersData parameters,
                                DoseData dosi, Mu_MuEn_Table mu_table, HistoryMap hist_map);

void kernel_host_seTLE( ParticlesData particles, VoxVolumeData vol,
                        COOHistoryMap coo_hist_map,
                        DoseData dosi,
                        Mu_MuEn_Table mu_table, ui32 nb_of_rays, f32 edep_th );

}

class VoxPhanIORTNav : public GGEMSPhantom
{
public:
    VoxPhanIORTNav();
    ~VoxPhanIORTNav() {}

    // Init
    void initialize( GlobalSimulationParameters params );
    // Tracking from outside to the phantom border
    void track_to_in( Particles particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out( Particles particles );

    void load_phantom_from_mhd ( std::string filename, std::string range_mat_name );

    void calculate_dose_to_phantom();
    void calculate_dose_to_water();
    
    void write( std::string filename = "dosimetry.mhd" );
    void set_materials( std::string filename );

    // TODO: TLE work only if the dosemap if the same than the phantom - JB
    //void set_doxel_size( f32 sizex, f32 sizey, f32 sizez );
    //void set_volume_of_interest( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax );

    void set_kerma_estimator( std::string kind );

    //void add_cylinder_objects( std::string filename, std::string mat_name );

    void export_density_map( std::string filename );
    void export_materials_map( std::string filename );
    void export_history_map( std::string filename );

private:

    VoxelizedPhantom m_phantom;
    Materials m_materials;
    CrossSections m_cross_sections;
    DoseCalculator m_dose_calculator;
    Mu_MuEn_Table m_mu_table;
    HistoryMap m_hist_map;
    COOHistoryMap m_coo_hist_map;

    bool m_check_mandatory();
    void m_init_mu_table();
    void m_compress_history_map();

    // Get the memory usage
    ui64 m_get_memory_usage();

    f32 m_dosel_size_x, m_dosel_size_y, m_dosel_size_z;
    f32 m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax;

    ui8 m_flag_TLE;

    GlobalSimulationParameters m_params;

    std::string m_materials_filename;

//
};

#endif
