// GGEMS Copyright (C) 2015

/*!
 * \file point_source.cuh
 * \brief Header of point source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Point source class
 *
 */

#ifndef POINT_SOURCE_CUH
#define POINT_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"
#include "ggems_source.cuh"
#include "prng.cuh"
#include "vector.cuh"

class GGEMSource;

// Sphere
class PointSource : public GGEMSSource
{
public:
    PointSource();
    ~PointSource();

    // Setting
    void set_position( f32 posx, f32 posy, f32 posz );
    void set_particle_type( std::string pname );
    void set_energy( f32 energy );

    // Getting
    f32xyz get_position();

    // Abstract from GGEMSSource (Mandatory funtions)
    void get_primaries_generator( Particles particles );
    void initialize( GlobalSimulationParametersData *h_params );

private:
    bool m_check_mandatory();

    GlobalSimulationParametersData *mh_params;

    f32 m_px, m_py, m_pz;
    f32 m_energy;
    ui8 m_particle_type;
};

#endif
