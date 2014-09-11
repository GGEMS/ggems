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

#ifndef POINT_SOURCE_CUH
#define POINT_SOURCE_CUH

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "../processes/particles.cuh"
#include "../maths/prng.cuh"
#include "../processes/constants.cuh"

// External function
__host__ __device__ void point_source_primary_generator(ParticleStack particles, unsigned int id,
                                                        float px, float py, float pz, float energy,
                                                        unsigned char type, unsigned int geom_id);

// Sphere
class PointSource {
    public:
        PointSource(float ox, float oy, float oz, float E, unsigned int val_seed,
                    std::string src_name, unsigned int geom_id);

        float px, py, pz, energy;
        unsigned int seed, geometry_id;
        std::string source_name;

    private:
};

#endif
