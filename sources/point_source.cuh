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

#include "ggems_vsource.cuh"
#include "global.cuh"

// External function
__host__ __device__ void point_source_primary_generator(ParticleStack particles, ui32 id,
                                                        f32 px, f32 py, f32 pz, f32 energy,
                                                        ui8 type, ui32 geom_id);
// Sphere
class PointSource : public GGEMSVSource {
    public:
        PointSource(f32 vpx, f32 vpy, f32 vpz, ui32 vseed, std::string vname, ui32 vgeom_id);
        PointSource();

        void set_position(f32 vpx, f32 vpy, f32 vpz);



        void get_primaries_generator();

        f32 px, py, pz;

        std::vector<f32> energy_hist, partpdec;

    private:
};

#endif
