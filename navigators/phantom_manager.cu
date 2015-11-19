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

#ifndef PHANTOMMANAGER_CU
#define PHANTOMMANAGER_CU

#include "phantom_manager.cuh"

PhantomManager::PhantomManager() {
    m_phantom_name = "";
}

void PhantomManager::set_phantom(VoxPhanImgNav &aPhantom) {
    m_vox_phan_img = aPhantom;
    m_phantom_name = "VoxPhanImgNav";
}

void PhantomManager::initialize(GlobalSimulationParameters params) {

    // Init source that including also data copy to GPU
    if (m_phantom_name == "VoxPhanImgNav") {
        m_vox_phan_img.initialize(params);
    }

    // Put others phantom here.

}

// Get list of materials
std::vector<std::string> PhantomManager::get_materials_list() {

    if (m_phantom_name == "VoxPhanImgNav") {
        m_vox_phan_img.get_materials_list();
    }

    // Put others phantom here.
}

// Get data that contains materials index
ui16* PhantomManager::get_data_materials_indices() {

    if (m_phantom_name == "VoxPhanImgNav") {
        m_vox_phan_img.get_data_materials_indices();
    }

    // Put others phantom here.

}

// Move particle to the phantom boundary
void PhantomManager::track_to_in(ParticleStack &particles_h, ParticleStack &particles_d) {
    if (m_phantom_name == "VoxPhanImgNav") {
        m_vox_phan_img.track_to_in(particles_h, particles_d);
    }

    // Put others phantom here.
}

std::string PhantomManager::get_phantom_name() {
    return m_phantom_name;
}

#endif


















