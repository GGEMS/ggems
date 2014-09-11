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

#ifndef POINT_SOURCE_CU
#define POINT_SOURCE_CU

#include "point_source.cuh"

// External function
__host__ __device__ void point_source_primary_generator(ParticleStack particles, unsigned int id,
                                                        float px, float py, float pz, float energy,
                                                        unsigned char type, unsigned int geom_id) {

    float phi = JKISS32(particles, id);
    float theta = JKISS32(particles, id);

    phi  *= gpu_twopi;
    theta = acosf(1.0f - 2.0f*theta);

    // set photons
    particles.E[id] = energy;
    particles.dx[id] = cosf(phi)*sinf(theta);
    particles.dy[id] = sinf(phi)*sinf(theta);
    particles.dz[id] = cosf(theta);
    particles.px[id] = px;
    particles.py[id] = py;
    particles.pz[id] = pz;
    particles.tof[id] = 0.0f;
    particles.endsimu[id] = DISABLED;
    particles.level[id] = PRIMARY;
    particles.pname[id] = type;
    particles.geometry_id = geom_id;
}


PointSource::PointSource(float ox, float oy, float oz, float E, unsigned int val_seed,
                         std::string src_name, unsigned int geom_id) {

    px = ox;
    py = oy;
    pz = oz;
    energy = E;
    source_name = src_name;
    seed = val_seed;
    geometry_id = geom_id;
}

#endif
