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
#include "voxelized.cuh"
#include "raytracing.cuh"
#include "vector.cuh"
#include "materials.cuh"
#include "photon.cuh"
#include "photon_navigator.cuh"

class VoxPhanImgNav {
    public:
        VoxPhanImgNav() {}
        ~VoxPhanImgNav() {}

        // Tracking from outside to the phantom broder
        void track_to_in(Particles particles);
        // Tracking inside the phantom until the phantom border
        void track_to_out(Particles particles, Materials materials, PhotonCrossSection photon_CS);
        
        void load_phantom_from_mhd(std::string, std::string);  
        
        // Init
        void initialize(GlobalSimulationParameters params);

        // Get list of materials
        std::vector<std::string> get_materials_list();
        // Get data that contains materials index
        ui16* get_data_materials_indices();
        // Get the size of data (nb of voxels)
        ui32 get_data_size();


    private:
    
        VoxelizedPhantom phantom;
    
        bool m_check_mandatory();       
        void m_copy_phantom_cpu2gpu();

        GlobalSimulationParameters m_params;

};

#endif
