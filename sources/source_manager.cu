// GGEMS Copyright (C) 2015

#ifndef SOURCEMANAGER_CU
#define SOURCEMANAGER_CU

#include "source_manager.cuh"

SourcesManager::SourcesManager() {
    m_source_name = "";
}

void SourcesManager::set_source(PointSource &aSource) {
    m_point_source = aSource;
    m_source_name = "PointSource";
}

void SourcesManager::initialize(GlobalSimulationParameters params) {

    // Init source that including also data copy to GPU
    if (m_source_name == "PointSource") {
        m_point_source.initialize(params);
    }

    // Put others sources here.

}

void SourcesManager::get_primaries_generator(ParticleStack particles) {

    // Fill the buffer of new particles
    if (m_source_name == "PointSource") {
        m_point_source.get_primaries_generator(particles);
    }

}

std::string SourcesManager::get_source_name() {
    return m_source_name;
}

#endif
