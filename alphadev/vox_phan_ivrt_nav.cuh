// GGEMS Copyright (C) 2017

/*!
 * \file vox_phan_img_nav.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 18 novembre 2015
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 */

#ifndef VOX_PHAN_IVRT_NAV_CUH
#define VOX_PHAN_IVRT_NAV_CUH

#include "global.cuh"
#include "ggems_phantom.cuh"
#include "voxelized.cuh"
#include "raytracing.cuh"
#include "vector.cuh"
#include "materials.cuh"
#include "photon.cuh"
#include "photon_navigator.cuh"
#include "cross_sections.cuh"
#include "transport_navigator.cuh"
#include "image_io.cuh"

// debug
#include "particles.cuh"

// For variance reduction (use in VRT for instance)
#define IMG_VRT_ANALOG   0
#define IMG_VRT_WOODCOCK 1
#define IMG_VRT_SVW      2 // Super Voxel Woodcock

// VoxPhanIVRTNav -> VPIVRTN
namespace VPIVRTN
{
__device__ void track_to_out_analog( ParticlesData *particles,
                              const VoxVolumeData<ui16> *vol,
                              const MaterialsData *materials,
                              const PhotonCrossSectionData *photon_CS_table,
                              const GlobalSimulationParametersData *parameters,
                              ui32 part_id );

__device__ void track_to_out_woodcock( ParticlesData *particles,
                                       const VoxVolumeData<ui16> *vol,
                                       const MaterialsData *materials,
                                       const PhotonCrossSectionData *photon_CS_table,
                                       const GlobalSimulationParametersData *parameters,
                                       ui32 part_id,
                                       f32* mumax_table );

__device__ void track_to_out_svw(ParticlesData *particles,
                                  const VoxVolumeData<ui16> *vol,
                                  const MaterialsData *materials,
                                  const PhotonCrossSectionData *photon_CS_table,
                                  const GlobalSimulationParametersData *parameters,
                                  ui32 part_id,
                                  f32* mumax_table,
                                  ui8 *mumax_index_table,
                                  ui32 nb_bins_sup_voxel,
                                  ui32xyzw nb_sup_voxel );

__global__ void kernel_device_track_to_in(  ParticlesData *particles, f32 xmin, f32 xmax,
                                            f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 geom_tolerance );

__global__ void kernel_device_track_to_out_analog( ParticlesData *particles,
                                            const VoxVolumeData<ui16> *vol,
                                            const MaterialsData *materials,
                                            const PhotonCrossSectionData *photon_CS_table,
                                            const GlobalSimulationParametersData *parameters );

__global__ void kernel_device_track_to_out_woodcock( ParticlesData *particles,
                                                     const VoxVolumeData<ui16> *vol,
                                                     const MaterialsData *materials,
                                                     const PhotonCrossSectionData *photon_CS_table,
                                                     const GlobalSimulationParametersData *parameters,
                                                     f32* mumax_table );

__global__ void kernel_device_track_to_out_svw( ParticlesData *particles,
                                                const VoxVolumeData<ui16> *vol,
                                                const MaterialsData *materials,
                                                const PhotonCrossSectionData *photon_CS_table,
                                                const GlobalSimulationParametersData *parameters,
                                                f32* mumax_table,
                                                ui8 *mumax_index_table,
                                                ui32 nb_bins_sup_voxel,
                                                ui32xyzw nb_sup_voxel );
}

class VoxPhanIVRTNav : public GGEMSPhantom
{
public:
    VoxPhanIVRTNav();
    ~VoxPhanIVRTNav() {}

    // Init
    void initialize( GlobalSimulationParametersData *h_params,  GlobalSimulationParametersData *d_params);
    // Tracking from outside to the phantom broder
    void track_to_in ( ParticlesData *d_particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out ( ParticlesData *d_particles );

    void load_phantom_from_mhd ( std::string filename, std::string range_mat_name );
    void set_materials( std::string filename );

    void set_vrt( std::string kind );

    // Set the super voxel resolution
    void set_nb_bins_sup_voxel( ui32 nb_bins_sup_voxel );

    AabbData get_bounding_box();

private:
    std::string m_materials_filename;

    VoxelizedPhantom m_phantom;
    Materials m_materials;
    CrossSections m_cross_sections;

    bool m_check_mandatory();

    // Return the memory usage
    ui64 m_get_memory_usage();

    f32 m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax;
    ui32 m_nb_bins_sup_voxel;
    ui32xyzw m_nb_sup_voxel;

    GlobalSimulationParametersData *mh_params;
    GlobalSimulationParametersData *md_params;

    ui8 m_flag_vrt;

    // Experimental (Woodcock tracking)
    void m_build_mumax_table();

    // Experimental (Super Voxel Woodcock tracking)
    void m_build_svw_mumax_table();

    f32* m_mumax_table;
    ui8* m_mumax_index_table;
};

#endif
