// GGEMS Copyright (C) 2017

/*!
 * \file ggems_phantom.cuh
 * \brief Header of the abstract phantom class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 17 december 2015
 *
 * Abstract class that handle every phantoms used in GGEMS
 *
 * v0.2: JB - Change all structs and remove CPU exec
 */

#ifndef GGEMS_PHANTOM_CUH
#define GGEMS_PHANTOM_CUH

#include "global.cuh"
#include "particles.cuh"

class GGEMSPhantom
{
public:
    GGEMSPhantom();
    virtual ~GGEMSPhantom() {}
    virtual void initialize( GlobalSimulationParametersData *h_params, GlobalSimulationParametersData *d_params ) = 0;
    virtual void track_to_in( ParticlesData *d_particles ) = 0;
    virtual void track_to_out( ParticlesData *d_particles ) = 0;

    void set_name ( std::string name );
    std::string get_name();

private:
    std::string m_phantom_name;

};

#endif
