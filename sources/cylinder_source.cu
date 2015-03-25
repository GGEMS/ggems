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

#ifndef CYLINDER_SOURCE_CU
#define CYLINDER_SOURCE_CU

#include "cylinder_source.cuh"

// External function
__host__ __device__ void cylinder_source_primary_generator(ParticleStack particles, ui32 id,
                                                        f32 px, f32 py, f32 pz, f32 rad,
                                                        f32 length, f32 energy,
                                                        ui8 type, ui32 geom_id) {

    // random position within the cylinder_source
    f32 theta = JKISS32(particles, id);
    theta *= gpu_twopi;
    
    f32 s = JKISS32(particles, id);
    f32 r = sqrt(s) * rad;
    
    f32 x = r*cosf(theta);
    f32 y = r*sinf(theta);
    f32 z = (JKISS32(particles, id) - 0.5) * length;
    
     // shift according to center of cylinder and translation
    x += px;
    y += py;
    z += pz;
    
    // random orientation
    f32 phi = JKISS32(particles, id);
    theta = JKISS32(particles, id);
    phi  *= gpu_twopi;
    theta = acosf(1.0f - 2.0f*theta);

    // set photons
    particles.E[id] = energy;
    particles.dx[id] = cosf(phi)*sinf(theta);
    particles.dy[id] = sinf(phi)*sinf(theta);
    particles.dz[id] = cosf(theta);
    particles.px[id] = x;
    particles.py[id] = y;
    particles.pz[id] = z;
    particles.tof[id] = 0.0f;
    particles.endsimu[id] = PARTICLE_ALIVE;
    particles.level[id] = PRIMARY;
    particles.pname[id] = type;
    particles.geometry_id[id] = geom_id;
}


CylinderSource::CylinderSource(f32 vpx, f32 vpy, f32 vpz, f32 vrad, f32 vlen, 
                               f32 vE, ui32 vseed, std::string vname, ui32 vgeom_id) {

    // Default parameters
    px = vpx; py = vpy; pz = vpz;
    rad = vrad;
    length = vlen;
    energy = vE;
    source_name = vname;
    seed = vseed;
    geometry_id = vgeom_id;
}

CylinderSource::CylinderSource() {

    // Default parameters
    px = 0.0f; py = 0.0f; pz = 0.0f;
    rad = 0.0f; length = 0.0f;
    energy = 60.0*keV;
    source_name = "Source01";
    seed = 10;
    geometry_id = 0;
}

// Setting function

void CylinderSource::set_position(f32 vpx, f32 vpy, f32 vpz) {
    px=vpx; py=vpy; pz=vpz;
}

void CylinderSource::set_radius(f32 vrad) {
    rad=vrad;
}

void CylinderSource::set_length(f32 vlen) {
    length=vlen;
}

void CylinderSource::set_energy(f32 venergy) {
    energy=venergy;
}

void CylinderSource::set_seed(ui32 vseed) {
    seed=vseed;
}

void CylinderSource::set_in_geometry(ui32 vgeometry_id) {
    geometry_id=vgeometry_id;
}

void CylinderSource::set_source_name(std::string vsource_name) {
    source_name=vsource_name;
}

#endif
