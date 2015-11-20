// GGEMS Copyright (C) 2015

/*!
 * \file phantom_manager.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 19 novembre 2015
 *
 *
 *
 */

#ifndef PHANTOMMANAGER_CUH
#define PHANTOMMANAGER_CUH

#include "global.cuh"
#include "vox_phan_img_nav.cuh"

class PhantomManager {
    public:
        PhantomManager();
        ~PhantomManager() {}

        void set_phantom(VoxPhanImgNav &aPhantom);
        void initialize(GlobalSimulationParameters params);

        // Get list of materials
        std::vector<std::string> get_materials_list();
        // Get data that contains materials index
        ui16* get_data_materials_indices();
        // Get the size of data
        ui32 get_data_size();

        void track_to_in(ParticleStack &particles_h, ParticleStack &particles_d);
        //void track_to_out();

        std::string get_phantom_name();

    private:
        VoxPhanImgNav m_vox_phan_img;
        std::string m_phantom_name;

};

#endif
