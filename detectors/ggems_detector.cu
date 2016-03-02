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

void GGEMSDetector::initialize(GlobalSimulationParameters params) {}

// Move particle to the phantom boundary
void GGEMSDetector::track_to_in(Particles particles) {}

// Track particle within the phantom
void GGEMSDetector::track_to_out(Particles particles) {}

void GGEMSDetector::digitizer(Particles particles) {}

std::string GGEMSDetector::get_detector_name() {
    return m_detector_name;
}

#endif


















