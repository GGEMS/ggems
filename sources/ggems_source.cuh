// GGEMS Copyright (C) 2015

/*!
 * \file ggems_source.cuh
 * \brief Header of the abstract source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Abstract class that handle every sources used in GGEMS
 *
 */

#ifndef GGEMS_SOURCE_CUH
#define GGEMS_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"

class GGEMSSource {
    public:
        GGEMSSource();
        virtual ~GGEMSSource() {}
        virtual void get_primaries_generator(Particles particles) = 0;
        virtual void initialize(GlobalSimulationParameters params) = 0;

        void set_name(std::string name);
        std::string get_name();

    private:
        std::string m_source_name;

};

#endif
