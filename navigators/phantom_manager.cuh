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
#include "materials.cuh"
#include "cross_sections.cuh"
#include "vox_phan_img_nav.cuh"

class PhantomManager {
    public:
        PhantomManager();
        ~PhantomManager() {}

        void set_phantom(VoxPhanImgNav &aPhantom);
        void initialize(GlobalSimulationParameters params);

        void track_to_in(Particles particles);
        void track_to_out(Particles particles);

        std::string get_phantom_name();

    private:
        VoxPhanImgNav m_vox_phan_img;
        std::string m_phantom_name;      

        MaterialManager m_materials_mng;
        CrossSectionsManager m_cross_sections_mng;

};

#endif
