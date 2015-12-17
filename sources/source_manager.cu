// GGEMS Copyright (C) 2015

/*!
 * \file source_manager.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef SOURCEMANAGER_CU
#define SOURCEMANAGER_CU

#include "source_manager.cuh"

void SourcesManager::set_source(GGEMSSource* aSource) {
    m_source = aSource;
}

void SourcesManager::initialize(GlobalSimulationParameters params) {
    m_source->initialize(params);
}

void SourcesManager::get_primaries_generator(Particles particles) {
    m_source->get_primaries_generator(particles);
}

std::string SourcesManager::get_name() {
    m_source->get_name();
}

#endif
