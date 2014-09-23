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

void cpu_photon_navigator(ParticleStack &particles, unsigned int part_id,
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
    unsigned int id_geom = particles.geometry_id[part_id];

    //if (part_id == 59520)
    //    printf("id %i pos %f %f %f dir %f %f %f E %f geom %i\n", part_id, pos.x, pos.y, pos.z,
    //           dir.x, dir.y, dir.z, particles.E[part_id], id_geom);

    // Get the material that compose this volume
    unsigned int adr_geom = geometry.ptr_objects[id_geom];
    unsigned int obj_type = (unsigned int)geometry.data_objects[adr_geom+ADR_OBJ_TYPE];
    unsigned int id_mat = 0;
    if (obj_type != VOXELIZED) {
        id_mat = (unsigned int)geometry.data_objects[adr_geom+ADR_OBJ_MAT_ID];
    } else {
        // TODO
        id_mat = 0;
    }

    //// Find next discrete interaction ///////////////////////////////////////

    float next_interaction_distance = FLT_MAX;
    unsigned char next_discrete_process = 0;
    unsigned int next_geometry_volume = id_geom;
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
        cross_section = Compton_CS_standard(materials, id_mat, particles.E[part_id]);
        interaction_distance = -log( JKISS32(particles, part_id) ) / cross_section;
        //if (part_id == 59520) printf("  Cpt dist %f\n", interaction_distance);
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

    // First check the boundary of the current volume
    if (obj_type == AABB) {

        // Read first the bounding box
        float xmin = geometry.data_objects[adr_geom+ADR_AABB_XMIN];
        float xmax = geometry.data_objects[adr_geom+ADR_AABB_XMAX];
        float ymin = geometry.data_objects[adr_geom+ADR_AABB_YMIN];
        float ymax = geometry.data_objects[adr_geom+ADR_AABB_YMAX];
        float zmin = geometry.data_objects[adr_geom+ADR_AABB_ZMIN];
        float zmax = geometry.data_objects[adr_geom+ADR_AABB_ZMAX];

        interaction_distance = hit_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax);
        //if (part_id == 59520) printf("  Cur Geom dist %e id %i (%f %f %f %f %f %f)\n",
        //                             interaction_distance, id_geom, xmin, xmax, ymin, ymax, zmin, zmax);

        if (interaction_distance <= next_interaction_distance) {
            next_interaction_distance = interaction_distance + EPSILON3; // overshoot
            next_discrete_process = GEOMETRY_BOUNDARY;
            next_geometry_volume = geometry.mother_node[id_geom];
        }

    } else if (obj_type == SPHERE) {
        // TODO

    } // else if VOXELIZED   ... etc.

    // Then check every child contains in this node
    unsigned int adr_node = geometry.ptr_nodes[id_geom];

    unsigned int offset_node = 0;
    unsigned int id_child_geom;
    while (offset_node < geometry.size_of_nodes[id_geom]) {

        // Child id
        id_child_geom = geometry.child_nodes[adr_node + offset_node];

        // Determine the type of the volume
        unsigned int adr_child_geom = geometry.ptr_objects[id_child_geom];
        unsigned int obj_child_type = (unsigned int)geometry.data_objects[adr_child_geom+ADR_OBJ_TYPE];

        // Get raytracing distance accordingly
        if (obj_child_type == AABB) {

            // Read first the bounding box
            float xmin = geometry.data_objects[adr_child_geom+ADR_AABB_XMIN];
            float xmax = geometry.data_objects[adr_child_geom+ADR_AABB_XMAX];
            float ymin = geometry.data_objects[adr_child_geom+ADR_AABB_YMIN];
            float ymax = geometry.data_objects[adr_child_geom+ADR_AABB_YMAX];
            float zmin = geometry.data_objects[adr_child_geom+ADR_AABB_ZMIN];
            float zmax = geometry.data_objects[adr_child_geom+ADR_AABB_ZMAX];

            // Ray/AABB raytracing
            interaction_distance = hit_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax);
            //if (part_id == 59520) printf("  Child Geom dist %e id %i (%f %f %f %f %f %f)\n",
            //                        interaction_distance, id_child_geom, xmin, xmax, ymin, ymax, zmin, zmax);

            if (interaction_distance <= next_interaction_distance) {
                next_interaction_distance = interaction_distance + EPSILON3; // overshoot
                next_discrete_process = GEOMETRY_BOUNDARY;
                next_geometry_volume = id_child_geom;
            }

        } else if (obj_child_type == SPHERE) {
            // do
        } // else if VOXELIZED   ... etc.

        ++offset_node;

    }

    //// Move particle //////////////////////////////////////////////////////

    // TODO
    // Compute the energy deposit position randomly along the path
    //if (parameters.dose_flag) {
        //float3 pos_edep = add_vector(photon.pos, scale_vector(photon.dir, next_interaction_distance*prng()));
    //}

    // Move the particle
    pos = f3_add(pos, f3_scale(dir, next_interaction_distance));
    //if (part_id == 59520) printf("  => mvt %e\n", next_interaction_distance);

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

        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    //float discrete_loss = 0.0f;

    if (next_discrete_process == PHOTON_COMPTON) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                 cutE
        SecParticle electron = Compton_SampleSecondaries_standard(particles, 0.0, part_id, parameters);

        // Local deposition if this photon was absorbed
        //if (particles.endsimu[part_id] == PARTICLE_DEAD) discrete_loss = particles.E[part_id];

        // Generate electron if need
        if (electron.endsimu == PARTICLE_ALIVE) {
            // TODO
        }

    }

    // Record this step if required
    if (history.record_flag == ENABLED) {
        history.cpu_record_a_step(particles, part_id);
    }


}



















#endif
