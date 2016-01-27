// GGEMS Copyright (C) 2015

/*!
 * \file ggems_detector.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 *
 *
 */

#ifndef GGEMSDETECTOR_CU
#define GGEMSDETECTOR_CU

#include "ggems_detector.cuh"

GGEMSDetector::GGEMSDetector()
: m_detector_name( "no_source" )
{
  ;
}

void GGEMSDetector::set_name(std::string name) {
    m_detector_name = name;
}

/*
void DetectorManager::set_detector(VoxPhanImgNav &aPhantom) {
    m_vox_phan_img = aPhantom;
    m_phantom_name = "VoxPhanImgNav";
}
*/

void GGEMSDetector::initialize(GlobalSimulationParameters params) {

    /*
    // Init phantom that including also data copy to GPU
    if (m_detector_name == "VoxPhanImgNav") {

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
*/
    // Put others phantom here.

}

// Move particle to the phantom boundary
void GGEMSDetector::track_to_in(Particles particles) {
/*
    if (m_phantom_name == "VoxPhanImgNav") {
        m_vox_phan_img.track_to_in(particles);
    }
*/
    // Put others phantom here.
}

// Track particle within the phantom
void GGEMSDetector::track_to_out(Particles particles) {
/*
    if (m_phantom_name == "VoxPhanImgNav") {
        m_vox_phan_img.track_to_out(particles, m_materials_mng.materials, m_cross_sections_mng.photon_CS);
    }
*/
}

void GGEMSDetector::digitizer()
{


}

std::string GGEMSDetector::get_detector_name() {
    return m_detector_name;
}

#endif


















