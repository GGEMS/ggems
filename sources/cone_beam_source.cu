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

#ifndef CONE_BEAM_SOURCE_CU
#define CONE_BEAM_SOURCE_CU

#include "cone_beam_source.cuh"

// External function
__host__ __device__ void cone_beam_source_primary_generator(ParticleStack particles, ui32 id,
                                                            f32 px, f32 py, f32 pz,
                                                            f32 rphi, f32 rtheta, f32 rpsi,
                                                            f32 aperture, f32 energy,
                                                            ui8 pname, ui32 geom_id) {

    // Get direction
    f32 phi = JKISS32(particles, id);
    f32 theta = JKISS32(particles, id);
    f32 val_aper = 1.0f - cosf(aperture);
    phi  *= gpu_twopi;
    theta = acosf(1.0f - val_aper*theta);

    f32 dx = cosf(phi)*sinf(theta);
    f32 dy = sinf(phi)*sinf(theta);
    f32 dz = cosf(theta);

    // Apply rotation
    f32xyz d = f3_rotate(make_f32xyz(dx, dy, dz), make_f32xyz(rphi, rtheta, rpsi));

    // set photons
    particles.E[id] = energy;
    particles.dx[id] = d.x;
    particles.dy[id] = d.y;
    particles.dz[id] = d.z;
    particles.px[id] = px;
    particles.py[id] = py;
    particles.pz[id] = pz;
    particles.tof[id] = 0.0f;
    particles.endsimu[id] = PARTICLE_ALIVE;
    particles.level[id] = PRIMARY;
    particles.pname[id] = pname;
    particles.geometry_id[id] = geom_id;
}


ConeBeamSource::ConeBeamSource() {

    // Default parameters
    px = 0.0f; py = 0.0f; pz = 0.0f;
    phi = 0.0f; theta = 0.0f; psi = 0.0f;
    aperture = 8.0f*deg;
    energy = 60.0*keV;
    source_name = "Source01";
    seed = 10;
    geometry_id = 0;

}

// Setting function

void ConeBeamSource::set_position(f32 vpx, f32 vpy, f32 vpz) {
    px=vpx; py=vpy; pz=vpz;
}

void ConeBeamSource::set_rotation(f32 vphi, f32 vtheta, f32 vpsi) {
    phi=vphi; theta=vtheta; psi=vpsi;
}

void ConeBeamSource::set_aperture(f32 vaperture) {
    aperture=vaperture;
}

void ConeBeamSource::set_energy(f32 venergy) {
    energy=venergy;
}

void ConeBeamSource::set_seed(ui32 vseed) {
    seed=vseed;
}

void ConeBeamSource::set_in_geometry(ui32 vgeometry_id) {
    geometry_id=vgeometry_id;
}

void ConeBeamSource::set_source_name(std::string vsource_name) {
    source_name=vsource_name;
}







#endif
