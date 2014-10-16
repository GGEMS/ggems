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

// CPU photon navigator
__host__ void cpu_photon_navigator(ParticleStack &particles, unsigned int part_id,
                          Scene geometry, MaterialsTable materials,
                          GlobalSimulationParameters parameters,
                          HistoryBuilder &history) {


    // Read position
    float3 pos;
    pos.x = particles.px[part_id];
    pos.y = particles.py[part_id];
    pos.z = particles.pz[part_id];

    // Read direction
    float3 dir;
    dir.x = particles.dx[part_id];
    dir.y = particles.dy[part_id];
    dir.z = particles.dz[part_id];

    // Get the current volume containing the particle
    unsigned int cur_id_geom = particles.geometry_id[part_id];

    // Get the material that compose this volume
    unsigned int id_mat = get_geometry_material(geometry, cur_id_geom);

    //// Find next discrete interaction ///////////////////////////////////////

    float next_interaction_distance = FLT_MAX;
    unsigned char next_discrete_process = 0;
    unsigned int next_geometry_volume = cur_id_geom;
    float interaction_distance;
    float cross_section;

    // If photoelectric
    if (parameters.physics_list[PHOTON_PHOTOELECTRIC]) {
        cross_section = PhotoElec_CS_standard(materials, id_mat, particles.E[part_id]);
        interaction_distance = -log( JKISS32(particles, part_id) ) / cross_section;
        if (interaction_distance < next_interaction_distance) {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_PHOTOELECTRIC;
        }
    }

    // If Compton
    if (parameters.physics_list[PHOTON_COMPTON]) {
        cross_section = Compton_CS_standard(materials, id_mat, particles.E[part_id]);
        interaction_distance = -log( JKISS32(particles, part_id) ) / cross_section;
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

    //// Get the next distance boundary volume /////////////////////////////////

    unsigned int hit_id_geom = 0;
    get_next_geometry_boundary(geometry, cur_id_geom, pos, dir, interaction_distance, hit_id_geom);
    if (interaction_distance <= next_interaction_distance) {
        next_interaction_distance = interaction_distance;
        next_discrete_process = GEOMETRY_BOUNDARY;
        next_geometry_volume = hit_id_geom;
    }

    //// Move particle //////////////////////////////////////////////////////

    // TODO
    // Compute the energy deposit position randomly along the path
    //if (parameters.dose_flag) {
        //float3 pos_edep = add_vector(photon.pos, scale_vector(photon.dir, next_interaction_distance*prng()));
    //}

    // Move the particle
    pos = f3_add(pos, f3_scale(dir, next_interaction_distance));

    // TODO
    //particles.tof[id] += gpu_speed_of_light * next_interaction_distance;

    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    particles.geometry_id[part_id] = next_geometry_volume;

    // Check world boundary
    float xmin = geometry.data_objects[ADR_AABB_XMIN]; // adr_world_geom = 0
    float xmax = geometry.data_objects[ADR_AABB_XMAX];
    float ymin = geometry.data_objects[ADR_AABB_YMIN];
    float ymax = geometry.data_objects[ADR_AABB_YMAX];
    float zmin = geometry.data_objects[ADR_AABB_ZMIN];
    float zmax = geometry.data_objects[ADR_AABB_ZMAX];

    // Stop simulation if out of the world
    if (   pos.x <= xmin || pos.x >= xmax
        || pos.y <= ymin || pos.y >= ymax
        || pos.z <= zmin || pos.z >= zmax) {

        particles.endsimu[part_id] = PARTICLE_DEAD;

        // Record this step if required
        if (history.record_flag == ENABLED) {
            history.cpu_record_a_step(particles, part_id);
        }

        //if (particles.E[part_id] == 0.5) printf("No Interaction\n");

        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    //float discrete_loss = 0.0f;

    if (next_discrete_process == PHOTON_COMPTON) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                 cutE
        SecParticle electron = Compton_SampleSecondaries_standard(particles, 0.0, part_id, parameters);

        // Debug
        //printf("id %i - pos %f %f %f - dir %f %f %f - Cmpt - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
        //                                                                 dir.x, dir.y, dir.z,
        //                                                                 cur_id_geom, next_geometry_volume);
    }

    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                              cutE
        SecParticle electron = PhotoElec_SampleSecondaries_standard(particles, materials, 0.0,
                                                                    id_mat, part_id, parameters);
        // Debug
        //printf("id %i - pos %f %f %f - dir %f %f %f - PE - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
        //                                                               dir.x, dir.y, dir.z,
        //                                                               cur_id_geom, next_geometry_volume);

    }

    /*
    if (next_discrete_process == GEOMETRY_BOUNDARY) {
        // Debug
        printf("id %i - pos %f %f %f - dir %f %f %f - Bnd - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
                                                                         dir.x, dir.y, dir.z,
                                                                         cur_id_geom, next_geometry_volume);

    }
    */


    // WARNING: drop energy for every "dead" particle (gamma + e-)

    // Local deposition if this photon was absorbed
    //if (particles.endsimu[part_id] == PARTICLE_DEAD) discrete_loss = particles.E[part_id];


    // Record this step if required
    if (history.record_flag == ENABLED) {
        history.cpu_record_a_step(particles, part_id);
    }


    // DEBUGING: phasespace
    if (next_geometry_volume == 0) {
        printf("%e %e %e %e %e %e %e\n", particles.E[part_id], pos.x, pos.y, pos.z, dir.x, dir.y, dir.z);
        particles.endsimu[part_id] = PARTICLE_DEAD;
        return;
    }


}



















#endif
