// GGEMS Copyright (C) 2015

/*!
 * \file source_manager.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */


#ifndef SOURCEMANAGER_CUH
#define SOURCEMANAGER_CUH

#include "global.cuh"
#include "ggems_vsource.cuh"

//class GGEMSVSource;

class SourcesManager {
    public:
        SourcesManager();
        ~SourcesManager() {}

        //void set_source(PointSource &aSource);
        void set_source(GGEMSVSource &aSource);
        void initialize(GlobalSimulationParameters params);

        void get_primaries_generator(Particles particles);


    private:
        //PointSource m_point_source;
        GGEMSVSource m_source;


};

#endif
