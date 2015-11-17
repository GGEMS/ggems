// GGEMS Copyright (C) 2015

#ifndef SOURCEMANAGER_CU
#define SOURCEMANAGER_CU

#include "source_manager.cuh"

SourcesManager::SourcesManager() {
    m_point_source == NULL;
    m_source_name = "";
}

SourcesManager::set_source(PointSource *aSource) {
    m_point_source = aSource;
    m_source_name = "PointSource";
}

SourcesManager::initialize(GlobalSimulationParameters params) {

    // Init source that including also data copy to GPU
    switch (m_source_name) {
        case "PointSource": {
            m_point_source->initialize(params);
            break;
        }
    }

}

SourcesManager::get_primaries_generator(ParticleStack particles) {

    // Fill the buffer of new particles
    switch (m_source_name) {
        case "PointSource": {
            m_point_source->get_primaries_generator(particles);
            break;
        }
    }

}

#endif
