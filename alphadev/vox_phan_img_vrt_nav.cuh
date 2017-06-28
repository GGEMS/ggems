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

#ifndef VOX_PHAN_IMG_VRT_NAV_CUH
#define VOX_PHAN_IMG_VRT_NAV_CUH

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

// VoxPhanImgNav -> VPIM
namespace VPIM
{
__device__ void track_to_out( ParticlesData *particles,
                              const VoxVolumeData<ui16> *vol,
                              const MaterialsData *materials,
                              const PhotonCrossSectionData *photon_CS_table,
                              const GlobalSimulationParametersData *parameters,
                              ui32 part_id );
__global__ void kernel_device_track_to_in(ParticlesData *particles, f32 xmin, f32 xmax,
                                            f32 ymin, f32 ymax, f32 zmin, f32 zmax, f32 geom_tolerance );
__global__ void kernel_device_track_to_out(ParticlesData *particles,
                                            const VoxVolumeData<ui16> *vol,
                                            const MaterialsData *materials,
                                            const PhotonCrossSectionData *photon_CS_table,
                                            const GlobalSimulationParametersData *parameters );
}

class VoxPhanImgVRTNav : public GGEMSPhantom
{
public:
    VoxPhanImgVRTNav();
    ~VoxPhanImgVRTNav() {}

    // Init
    void initialize( GlobalSimulationParametersData *h_params,  GlobalSimulationParametersData *d_params);
    // Tracking from outside to the phantom broder
    void track_to_in ( ParticlesData *d_particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out ( ParticlesData *d_particles );

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

    GlobalSimulationParametersData *mh_params;
    GlobalSimulationParametersData *md_params;
};

#endif
