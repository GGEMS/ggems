// GGEMS Copyright (C) 2015

/*!
 * \file phasespace_source.cuh
 * \brief Header of the phasespace source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 15 mars 2016
 *
 * Phasespace class
 *
 * UNDER CONSTRUCTION
 *
 */

#ifndef PHASESPACE_SOURCE_CUH
#define PHASESPACE_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"
#include "ggems_source.cuh"
#include "prng.cuh"
#include "phasespace_io.cuh"
#include "fun.cuh"
#include "vector.cuh"

struct PhSpTransform
{
    f32 *tx, *ty, *tz;
    f32 *rx, *ry, *rz;
    f32 *sx, *sy, *sz;

    f32 *cdf;

    ui32 nb_sources;
};

class GGEMSource;

namespace PHSPSRC
{
__host__ __device__ void phsp_source(ParticlesData particles_data,
                                     PhaseSpaceData phasespace, PhSpTransform transform, ui32 id );
__global__ void phsp_point_source( ParticlesData particles_data,
                                   PhaseSpaceData phasespace, PhSpTransform transform );
}

// PhaseSpace source
class PhaseSpaceSource : public GGEMSSource
{
public:
    PhaseSpaceSource();
    ~PhaseSpaceSource();

    // Setting    
    void set_translation( f32 tx, f32 ty, f32 tz );
    void set_rotation( f32 aroundx, f32 aroundy, f32 aroundz );
    //void set_scaling( f32 sx, f32 sy, f32 sz );
    void set_max_number_of_particles( ui32 nb_part_max );

    // Main
    void set_phasespace_file( std::string filename );
    void set_transformation_file( std::string filename );

    // Abstract from GGEMSSource (Mandatory funtions)
    void get_primaries_generator( Particles particles );
    void initialize( GlobalSimulationParameters params );

private:
    std::string m_phasespace_file;
    std::string m_transformation_file;
    void m_load_phasespace_file();
    void m_load_transformation_file();

    bool m_check_mandatory();
    void m_transform_allocation( ui32 nb_sources );
    void m_skip_comment(std::istream & is);

    GlobalSimulationParameters m_params;
    PhSpTransform m_transform;    
    PhaseSpaceData m_phasespace;    
    i32 m_nb_part_max;

};

#endif
