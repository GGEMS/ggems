// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_img_nav.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef VOX_PHAN_IMG_NAV_CUH
#define VOX_PHAN_IMG_NAV_CUH

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

// debug
#include "particles.cuh"

// VoxPhanImgNav -> VPIN
namespace VPIN
{
__host__ __device__ void track_to_out (ParticlesData &particles,
                                       const VoxVolumeData<ui16> &vol,
                                       const MaterialsTable &materials,
                                       const PhotonCrossSectionTable &photon_CS_table,
                                       const GlobalSimulationParametersData &parameters,
                                       ui32 part_id );
__global__ void kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                                            f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 geom_tolerance );
__global__ void kernel_device_track_to_out ( ParticlesData particles,
                                             const VoxVolumeData<ui16> vol,
                                             const MaterialsTable materials,
                                             const PhotonCrossSectionTable photon_CS_table,
                                             const GlobalSimulationParametersData parameters );
void kernel_host_track_to_in (ParticlesData &particles, f32 xmin, f32 xmax,
                              f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 geom_tolerance, ui32 id );
void kernel_host_track_to_out (ParticlesData &particles,
                               const VoxVolumeData<ui16> &vol,
                               const MaterialsTable &materials,
                               const PhotonCrossSectionTable &photon_CS_table,
                               const GlobalSimulationParametersData &parameters, ui32 id );
}

class VoxPhanImgNav : public GGEMSPhantom
{
public:
    VoxPhanImgNav();
    ~VoxPhanImgNav() {}

    // Init
    void initialize ( GlobalSimulationParameters params );
    // Tracking from outside to the phantom broder
    void track_to_in ( Particles particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out ( Particles particles );

    void load_phantom_from_mhd ( std::string filename, std::string range_mat_name );    
    void set_materials( std::string filename );

    AabbData get_bounding_box();

private:   
    std::string m_materials_filename;

    VoxelizedPhantom m_phantom;
    Materials m_materials;
    CrossSections m_cross_sections;

    bool m_check_mandatory();

    // Return the memory usage
    ui64 m_get_memory_usage();

    GlobalSimulationParameters m_params;
};

#endif
