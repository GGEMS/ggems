// GGEMS Copyright (C) 2017

/*!
 * \file phasespace_source.cuh
 * \brief Header of the phasespace source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 15 mars 2016
 *
 * Phasespace class
 *
 * v0.2: JB - Change all structs and remove CPU exec
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
#include "primitives.cuh"

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
__host__ __device__ void phsp_source(ParticlesData *particles_data,
                                     const PhaseSpaceData *phasespace,
                                     PhSpTransform transform, ui32 id );
__global__ void phsp_point_source(ParticlesData *particles_data,
                                   const PhaseSpaceData *phasespace,
                                   PhSpTransform transform );
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
    void set_scaling( f32 sx, f32 sy, f32 sz );
    void set_max_number_of_particles( ui32 nb_part_max );    

    // Main
    void set_phasespace_file( std::string filename );
    void set_transformation_file( std::string filename );

    // Update
    void update_phasespace_file( std::string filename );

    // Abstract from GGEMSSource (Mandatory funtions)
    void get_primaries_generator( ParticlesData *d_particles );
    void initialize( GlobalSimulationParametersData *h_params );

private:
    std::string m_phasespace_file;
    std::string m_transformation_file;
    void m_load_phasespace_file();
    void m_load_transformation_file();
    void m_copy_phasespace_to_gpu();    
    void m_free_phasespace_to_gpu();

    bool m_check_mandatory();
    void m_transform_allocation( ui32 nb_sources );
    void m_skip_comment(std::istream & is);

    GlobalSimulationParametersData *mh_params;
    PhSpTransform *mh_transform;
    PhSpTransform *md_transform;
    PhaseSpaceData *mh_phasespace;
    PhaseSpaceData *md_phasespace;
    i32 m_nb_part_max;

    f32 m_rx, m_ry, m_rz;
    f32 m_tx, m_ty, m_tz;
    f32 m_sx, m_sy, m_sz;

};

#endif
