// GGEMS Copyright (C) 2015

/*!
 * \file geom_source.cuh
 * \brief Header of the geom source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 9 mars 2016
 *
 * Geom source class
 *
 */

#ifndef GEOM_SOURCE_CUH
#define GEOM_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"
#include "ggems_source.cuh"
#include "prng.cuh"
#include "vector.cuh"
#include "fun.cuh"

struct Spectrum
{
    f32 *energy_h;          // Energy spectrum of the source on the host (CPU)
    f32 *energy_d;          // Energy spectrum of the source on the device (GPU)
    f32 *cdf_h;             // CDF of the source on the host (CPU)
    f32 *cdf_d;             // CDF of the source on the device (GPU)
    ui32 nb_of_energy_bins; // Number of the bins in the energy spectrum
};

struct aSource
{
    f32xyz pos;
    f32xyz rot;
    f32xyz length;
    f32 radius;
};

class GGEMSource;

namespace GEOMSRC
{
__host__ __device__ f32 get_energy( ParticlesData particles_data, f32 *energy, f32 *cdf, ui32 nb_bins, ui32 id );
__host__ __device__ void point_source (ParticlesData particles_data,
                                       f32xyz pos, f32 *energy, f32 *cdf, ui32 nb_bins, ui8 ptype, ui32 id);
__global__ void kernel_point_source (ParticlesData particles_data,
                                     f32xyz pos, f32 *energy, f32 *cdf, ui32 nb_bins, ui8 ptype );
}

// Geom source
class GeomSource : public GGEMSSource
{
public:
    GeomSource();
    ~GeomSource();

    // Setting
    void set_shape( std::string shape_name, std::string shape_mode );
    void set_position( f32 posx, f32 posy, f32 posz );
    void set_rotation( f32 aroundx, f32 aroundy, f32 aroundz );
    void set_length( f32 alongx, f32 alongy, f32 alongz );
    void set_radius( f32 radius );

    void set_particle_type( std::string pname );
    void set_mono_energy( f32 energy );
    void set_energy_spectrum( std::string filename );

    // Abstract from GGEMSSource (Mandatory funtions)
    void get_primaries_generator( Particles particles );
    void initialize( GlobalSimulationParameters params );

private:
    bool m_check_mandatory();

    GlobalSimulationParameters m_params;

    std::string m_shape;
    std::string m_shape_mode;
    f32xyz m_pos;
    f32xyz m_rot;
    f32xyz m_length;
    f32 m_radius;
    Spectrum *m_spectrum;
    ui8 m_particle_type;

};

#endif
