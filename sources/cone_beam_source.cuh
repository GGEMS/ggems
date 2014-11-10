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

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "particles.cuh"
#include "prng.cuh"
#include "constants.cuh"

// External function
__host__ __device__ void cone_beam_source_primary_generator(ParticleStack particles, unsigned int id,
                                                        float px, float py, float pz,
                                                        float rphi, float rtheta, float rpsi,
                                                        float aperture, float energy,
                                                        unsigned char pname, unsigned int geom_id);

// Sphere
class ConeBeamSource {
    public:
        ConeBeamSource();
        void set_position(float vpx, float vpy, float vpz);
        void set_rotation(float vphi, float vtheta, float vpsi);
        void set_aperture(float vaperture);
        void set_energy(float venergy);
        void set_seed(unsigned int vseed);
        void set_in_geometry(unsigned int vgeometry_id);
        void set_source_name(std::string vsource_name);

        float px, py, pz;
        float phi, theta, psi;
        float aperture, energy;
        unsigned int seed, geometry_id;
        std::string source_name;

    private:

};

#endif
