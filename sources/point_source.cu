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
                                                        f32 px, f32 py, f32 pz, f32 energy,
                                                        unsigned char type, unsigned int geom_id) {

    f32 phi = JKISS32(particles, id);
    f32 theta = JKISS32(particles, id);

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
    particles.endsimu[id] = PARTICLE_ALIVE;
    particles.level[id] = PRIMARY;
    particles.pname[id] = type;
    particles.geometry_id[id] = geom_id;
}


PointSource::PointSource() {

    // Default parameters
    px = 0.0f; py = 0.0f; pz = 0.0f;
    energy = 60.0*keV;
    source_name = "Source01";
    seed = 10;
    geometry_id = 0;
}

// Setting function

void PointSource::set_position(f32 vpx, f32 vpy, f32 vpz) {
    px=vpx; py=vpy; pz=vpz;
}

void PointSource::set_energy(f32 venergy) {
    energy=venergy;
}

void PointSource::set_seed(unsigned int vseed) {
    seed=vseed;
}

void PointSource::set_in_geometry(unsigned int vgeometry_id) {
    geometry_id=vgeometry_id;
}

void PointSource::set_source_name(std::string vsource_name) {
    source_name=vsource_name;
}

#endif
