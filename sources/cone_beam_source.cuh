// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef CONE_BEAM_SOURCE_CUH
#define CONE_BEAM_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"
#include "prng.cuh"
#include "constants.cuh"

// External function
__host__ __device__ void cone_beam_source_primary_generator(ParticleStack particles, unsigned int id,
                                                        f32 px, f32 py, f32 pz,
                                                        f32 rphi, f32 rtheta, f32 rpsi,
                                                        f32 aperture, f32 energy,
                                                        unsigned char pname, unsigned int geom_id);

// Sphere
class ConeBeamSource {
    public:
        ConeBeamSource();
        void set_position(f32 vpx, f32 vpy, f32 vpz);
        void set_rotation(f32 vphi, f32 vtheta, f32 vpsi);
        void set_aperture(f32 vaperture);
        void set_energy(f32 venergy);
        void set_seed(unsigned int vseed);
        void set_in_geometry(unsigned int vgeometry_id);
        void set_source_name(std::string vsource_name);

        f32 px, py, pz;
        f32 phi, theta, psi;
        f32 aperture, energy;
        unsigned int seed, geometry_id;
        std::string source_name;

    private:

};

#endif
