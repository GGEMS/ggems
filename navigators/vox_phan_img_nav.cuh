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
#include "image_reader.cuh"
#include "transport_navigator.cuh"

// debug
#include "particles.cuh"

// VoxPhanImgNav -> VPIN
namespace VPIN
{
__host__ __device__ void track_to_out ( ParticlesData &particles,
                                        VoxVolumeData vol,
                                        MaterialsTable materials,
                                        PhotonCrossSectionTable photon_CS_table,
                                        GlobalSimulationParametersData parameters,
                                        ui32 part_id );
__global__ void kernel_device_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
        f32 ymin, f32 ymax, f32 zmin, f32 zmax );
__global__ void kernel_device_track_to_out ( ParticlesData particles,
        VoxVolumeData vol,
        MaterialsTable materials,
        PhotonCrossSectionTable photon_CS_table,
        GlobalSimulationParametersData parameters );
void kernel_host_track_to_in ( ParticlesData particles, f32 xmin, f32 xmax,
                               f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 id );
void kernel_host_track_to_out ( ParticlesData particles,
                                VoxVolumeData vol,
                                MaterialsTable materials,
                                PhotonCrossSectionTable photon_CS_table,
                                GlobalSimulationParametersData parameters, ui32 id );
}

class VoxPhanImgNav : public GGEMSPhantom
{
public:
    VoxPhanImgNav() : m_elements_filename( "" ), m_materials_filename( "" ) {}
    ~VoxPhanImgNav() {}

    // Init
    void initialize ( GlobalSimulationParameters params );
    // Tracking from outside to the phantom broder
    void track_to_in ( Particles particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out ( Particles particles );

    void load_phantom_from_mhd ( std::string filename, std::string range_mat_name );
    void set_elements( std::string filename );
    void set_materials( std::string filename );

private:
    std::string m_elements_filename;
    std::string m_materials_filename;

    VoxelizedPhantom m_phantom;
    Materials m_materials;
    CrossSections m_cross_sections;

    bool m_check_mandatory();

    GlobalSimulationParameters m_params;
};

#endif
