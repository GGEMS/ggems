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
    m_phantom_name = m_vox_phan_img.get_name();
}

void PhantomManager::set_phantom(VoxPhanDosi &aPhantom) {
    m_vox_phan_dosi = aPhantom;
    m_phantom_name = m_vox_phan_dosi.get_name();
}

void PhantomManager::initialize(GlobalSimulationParameters params) {

    // Init materials data base
    m_materials_mng.load_elements_database();
    m_materials_mng.load_materials_database();

    // Init phantom that including also data copy to GPU
    if (m_phantom_name == m_vox_phan_img.get_name()) {

        /// Material handling ////////////////////////////
      
        // Build material data based on geometry
        m_materials_mng.add_materials_and_update_indices(m_vox_phan_img.get_materials_list(),
                                                         m_vox_phan_img.get_data_materials_indices(),
                                                         m_vox_phan_img.get_data_size());

        // Init material
        m_materials_mng.initialize(params);

        // Init CS
        m_cross_sections_mng.initialize(m_materials_mng.materials, params);

        // Init the phantom
        m_vox_phan_img.initialize(params);
    }
    else if (m_phantom_name == m_vox_phan_dosi.get_name()) {

        /// Material handling ////////////////////////////
      
        // Build material data based on geometry
        m_materials_mng.add_materials_and_update_indices(m_vox_phan_dosi.get_materials_list(),
                                                         m_vox_phan_dosi.get_data_materials_indices(),
                                                         m_vox_phan_dosi.get_data_size());

        // Init material
        m_materials_mng.initialize(params);

        // Init CS
        m_cross_sections_mng.initialize(m_materials_mng.materials, params);

        // Init the phantom
        m_vox_phan_dosi.initialize(params);
    }
    
    
    // Put others phantom here.

}

// Move particle to the phantom boundary
void PhantomManager::track_to_in(Particles particles) {

    if (m_phantom_name == m_vox_phan_img.get_name()) {
        m_vox_phan_img.track_to_in(particles);
    }
    else if (m_phantom_name == m_vox_phan_dosi.get_name()) {
        m_vox_phan_dosi.track_to_in(particles);
    }
    // Put others phantom here.
}

// Track particle within the phantom
void PhantomManager::track_to_out(Particles particles) {

    if (m_phantom_name == m_vox_phan_img.get_name()) {
        m_vox_phan_img.track_to_out(particles, m_materials_mng.materials, m_cross_sections_mng.photon_CS);
    }
    else if (m_phantom_name == m_vox_phan_dosi.get_name()) {
        m_vox_phan_dosi.track_to_out(particles, m_materials_mng.materials, m_cross_sections_mng);
    }

}

std::string PhantomManager::get_phantom_name() {
    return m_phantom_name;
}

#endif


















