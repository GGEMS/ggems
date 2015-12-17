// GGEMS Copyright (C) 2015

/*!
 * \file source_manager.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * \todo This class must be a singleton
 *
 */


#ifndef SOURCEMANAGER_CUH
#define SOURCEMANAGER_CUH

#include "global.cuh"
#include "ggems_source.cuh"

//class GGEMSVSource;

class SourcesManager {
    public:
        SourcesManager() {}
        ~SourcesManager() {}

        void set_source(GGEMSSource* aSource);
        void initialize(GlobalSimulationParameters params);
        void get_primaries_generator(Particles particles);
        std::string get_name();


    private:        
        GGEMSSource* m_source;


};

#endif
