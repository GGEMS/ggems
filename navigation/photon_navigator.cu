// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef PHOTON_NAVIGATOR_CU
#define PHOTON_NAVIGATOR_CU

#include "photon_navigator.cuh"

/*
void cpu_photon_navigator(ParticleBuilder particles, unsigned int part_id,
                          GeometryBuilder geometry, MaterialBuilder materials,
                          SimulationParameters parameters) {


    // Read position
    float3 pos;
    pos.x = particles.stack.px[part_id];
    pos.y = particles.stack.py[part_id];
    pos.z = particles.stack.pz[part_id];

    // Read direction
    float3 dir;
    dir.x = particles.stack.dx[part_id];
    dir.y = particles.stack.dy[part_id];
    dir.z = particles.stack.dz[part_id];

    // Get the current volume containing the particle
    unsigned int id_geom = particles.stack.geometry_id[part_id];

    // Get the material that compose this volume
    unsigned int adr_geom = geometry.World.ptr_objects[id_geom];
    unsigned int obj_type = (unsigned int)geometry.World.data_objects[adr_geom+ADR_OBJ_TYPE];
    if (obj_type != VOXELIZED) {
        unsigned int id_mat = (unsigned int)geometry.World.data_objects[adr_geom+ADR_OBJ_MAT_ID];
    } else {
        // TODO
        unsigned int id_mat = 0;
    }

    //// Find next discrete interaction ///////////////////////////////////////

    float next_interaction_distance = FLT_MAX;
    unsigned char next_discrete_process = 0;
    float interaction_distance;
    float cross_section;

    // If photoelectric
    if (parameters.physics_list[PHOTON_PHOTOELECTRIC]) {
        // TODO
        //cross_section = PhotoElec_CS_Standard(materials, mat, photon.E);
        //interaction_distance = -log(prng()) / cross_section;
        //if (interaction_distance < next_interaction_distance) {
        //    next_interaction_distance = interaction_distance;
        //    next_discrete_process = PHOTON_PHOTOELECTRIC;
        //}
    }

    // If Compton
    if (parameters.physics_list[PHOTON_COMPTON]) {
        cross_section = Compton_CS_Standard(materials, mat, photon.E);
        interaction_distance = -log(prng()) / cross_section;
        if (interaction_distance < next_interaction_distance) {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_COMPTON;
        }
    }

    // If Rayleigh
    if (parameters.physics_list[PHOTON_RAYLEIGH]) {
        // TODO
        //cross_section = Rayleigh_CS_Standard(materials, mat, photon.E);
        //interaction_distance = -log(prng()) / cross_section;
        //if (interaction_distance < next_interaction_distance) {
        //    next_interaction_distance = interaction_distance;
        //    next_discrete_process = PHOTON_RAYLEIGHT;
        //}
    }

    // TODO
    // Distance to the next voxel boundary (raycasting)
    //interaction_distance = get_boundary_voxel_by_raycasting(index_phantom, photon.pos,
    //                                                        photon.dir, voxel_size);
    //if (interaction_distance < next_interaction_distance) {
    //    next_interaction_distance = interaction_distance + EPS; // Overshoot
    //    next_discrete_process = PHOTON_BOUNDARY_VOXEL;
    //}

    //// Move particle //////////////////////////////////////////////////////

    // TODO
    // Compute the energy deposit position randomly along the path
    //if (parameters.dose_flag) {
        //float3 pos_edep = add_vector(photon.pos, scale_vector(photon.dir, next_interaction_distance*prng()));
    //}

    // Move the particle
    pos = f3_add(pos, f3_scale(dir, next_interaction_distance));

    // TODO
    // Stop simulation if out of phantom or no more energy
    //if (   photon.pos.x <= 0 || photon.pos.x >= (phantom.m_nx * phantom.m_spacing_x)
    //    || photon.pos.y <= 0 || photon.pos.y >= (phantom.m_ny * phantom.m_spacing_y)
    //    || photon.pos.z <= 0 || photon.pos.z >= (phantom.m_nz * phantom.m_spacing_z)) {
    //    photon.endsimu = 1;                     // stop the simulation
    //    return;
    //}

}



*/



















#endif
