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
#include "ggems_vphantom.cuh"
#include "voxelized.cuh"
#include "raytracing.cuh"
#include "vector.cuh"
#include "materials.cuh"
#include "photon.cuh"

class VoxPhanImgNav : public GGEMSVPhantom {
    public:
        VoxPhanImgNav() {}
        ~VoxPhanImgNav() {}

        // Tracking from outside to the phantom broder
        void track_to_in(ParticleStack &particles_h, ParticleStack &particles_d);
        // Tracking inside the phantom until the phantom border
        void track_to_out();
        // Init
        void initialize(GlobalSimulationParameters params);
        // Get list of materials
        std::vector<std::string> get_materials_list();
        // Get data that contains materials index
        ui16* get_data_materials_indices();
        // Get the size of data (nb of voxels)
        ui32 get_data_size();

        Voxelized phantom;

    private:
        bool m_check_mandatory();
        void m_copy_parameters_cpu2gpu();
        void m_copy_phantom_cpu2gpu();

        VoxVolume m_vox_vol_d;
        GlobalSimulationParameters m_params_h;
        GlobalSimulationParameters m_params_d;

};

#endif
