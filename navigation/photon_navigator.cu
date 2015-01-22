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
__host__ void cpu_photon_navigator(ParticleStack &particles, ui32 part_id,
                          Scene geometry, MaterialsTable materials,
                          PhotonCrossSectionTable photon_CS_table,
                          GlobalSimulationParameters parameters,
                          Singles &singles,
                          HistoryBuilder &history) {

    // Read position
    f64xyz pos;
    pos.x = particles.px[part_id];
    pos.y = particles.py[part_id];
    pos.z = particles.pz[part_id];

    // Read direction
    f64xyz dir;
    dir.x = particles.dx[part_id];
    dir.y = particles.dy[part_id];
    dir.z = particles.dz[part_id];

    // Get the current volume containing the particle
    ui32 cur_id_geom = particles.geometry_id[part_id];

    // Get the material that compose this volume
    ui32 id_mat = get_geometry_material(geometry, cur_id_geom, pos);

#ifdef DEBUG
    printf("  begin %i\n", part_id);
    printf("     Cur id geom %i mat %i\n", cur_id_geom, id_mat);
    printf("     InitPos %f %f %f\n", pos.x, pos.y, pos.z);
    printf("     Energy %f\n", particles.E[part_id]);
#endif

    //// Find next discrete interaction ///////////////////////////////////////

    f64 next_interaction_distance = F64_MAX;
    ui8 next_discrete_process = 0;
    ui32 next_geometry_volume = cur_id_geom;
    f64 interaction_distance;
    f64 cross_section;

    // Search the energy index to read CS
    ui32 E_index = binary_search(particles.E[part_id], photon_CS_table.E_bins,
                                 photon_CS_table.nb_bins);

    // TODO if E_index = 0?
    assert(E_index != 0);
    /////////////////////

    //printf("Before CS\n");

    // If photoelectric
    if (parameters.physics_list[PHOTON_PHOTOELECTRIC]) {
        cross_section = get_CS_from_table(photon_CS_table.E_bins, photon_CS_table.Photoelectric_Std_CS,
                                          particles.E[part_id], E_index, id_mat, photon_CS_table.nb_bins);
        interaction_distance = -log( JKISS32(particles, part_id) ) / cross_section;
#ifdef DEBUG
        printf(" Photoelectric: CS %e dist %e\n", cross_section, interaction_distance);
#endif
        if (interaction_distance < next_interaction_distance) {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_PHOTOELECTRIC;
        }
        //if (cur_id_geom==1) printf("E %e CS %e\n", particles.E[part_id], cross_section);
    }

    // If Compton
    if (parameters.physics_list[PHOTON_COMPTON]) {
        cross_section = get_CS_from_table(photon_CS_table.E_bins, photon_CS_table.Compton_Std_CS,
                                          particles.E[part_id], E_index, id_mat, photon_CS_table.nb_bins);
        interaction_distance = -log( JKISS32(particles, part_id) ) / cross_section;
#ifdef DEBUG
        printf(" Compton: CS %e dist %e\n", cross_section, interaction_distance);
#endif
        if (interaction_distance < next_interaction_distance) {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_COMPTON;
        }
    }

    // If Rayleigh
    if (parameters.physics_list[PHOTON_RAYLEIGH]) {
        cross_section = get_CS_from_table(photon_CS_table.E_bins, photon_CS_table.Rayleigh_Lv_CS,
                                          particles.E[part_id], E_index, id_mat, photon_CS_table.nb_bins);
        interaction_distance = -log( JKISS32(particles, part_id) ) / cross_section;
#ifdef DEBUG
        printf(" Rayleigh: CS %e dist %e\n", cross_section, interaction_distance);
#endif
        if (interaction_distance < next_interaction_distance) {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_RAYLEIGH;
        }

    }

    //// Get the next distance boundary volume /////////////////////////////////

    //printf("Before geom\n");

    ui32 hit_id_geom = 0;
    get_next_geometry_boundary(geometry, cur_id_geom, pos, dir, interaction_distance, hit_id_geom);
#ifdef DEBUG
        printf(" Geom: dist %e\n", interaction_distance);
#endif
    if (interaction_distance <= next_interaction_distance) {
        next_interaction_distance = interaction_distance + EPSILON3; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
        next_geometry_volume = hit_id_geom;        
    }

    //// Move particle //////////////////////////////////////////////////////

    //printf("Move particle\n");

    // TODO
    // Compute the energy deposit position randomly along the path
    //if (parameters.dose_flag) {
        //f32xyz pos_edep = add_vector(photon.pos, scale_vector(photon.dir, next_interaction_distance*prng()));
    //}

    // Move the particle
    pos = fxyz_add(pos, fxyz_scale(dir, next_interaction_distance));

    // TODO
    //particles.tof[id] += gpu_speed_of_light * next_interaction_distance;

    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    particles.geometry_id[part_id] = next_geometry_volume;

    // Check world boundary
    f64 xmin = geometry.data_objects[ADR_AABB_XMIN]; // adr_world_geom = 0
    f64 xmax = geometry.data_objects[ADR_AABB_XMAX];
    f64 ymin = geometry.data_objects[ADR_AABB_YMIN];
    f64 ymax = geometry.data_objects[ADR_AABB_YMAX];
    f64 zmin = geometry.data_objects[ADR_AABB_ZMIN];
    f64 zmax = geometry.data_objects[ADR_AABB_ZMAX];

    // Stop simulation if out of the world
    if (!test_point_AABB(pos, xmin, xmax, ymin, ymax, zmin, zmax)) {

        particles.endsimu[part_id] = PARTICLE_DEAD;

        // Record this step if required
        if (history.record_flag == ENABLED) {
            history.cpu_record_a_step(particles, part_id);
        }

        //if (particles.E[part_id] == 0.5) printf("No Interaction\n");

        return;
    }

//    // Stop simulation if out of the world
//    if (   pos.x <= xmin || pos.x >= xmax
//        || pos.y <= ymin || pos.y >= ymax
//        || pos.z <= zmin || pos.z >= zmax) {

//        particles.endsimu[part_id] = PARTICLE_DEAD;

//        // Record this step if required
//        if (history.record_flag == ENABLED) {
//            history.cpu_record_a_step(particles, part_id);
//        }

//        //if (particles.E[part_id] == 0.5) printf("No Interaction\n");

//        return;
//    }

    //// Apply discrete process //////////////////////////////////////////////////

    f32 discrete_loss = 0.0f;
    SecParticle electron;
    electron.E = 0;

#ifdef DEBUG
    printf("     Dist %f NextVol %i pos %f %f %f ", next_interaction_distance, next_geometry_volume, pos.x, pos.y, pos.z);
#endif

    if (next_discrete_process == PHOTON_COMPTON) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                 cutE
        electron = Compton_SampleSecondaries_standard(particles, 0.0, part_id, parameters);



        // Debug
        //printf("id %i - pos %f %f %f - dir %f %f %f - Cmpt - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
        //                                                                 dir.x, dir.y, dir.z,
        //                                                                 cur_id_geom, next_geometry_volume);
#ifdef DEBUG
        printf(" Compton\n");
#endif
    }

    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                                               cutE
        electron = Photoelec_SampleSecondaries_standard(particles, materials, photon_CS_table,
                                                                    E_index, 0.0, id_mat, part_id, parameters);

        // Debug
        //printf("id %i - pos %f %f %f - dir %f %f %f - PE - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
        //                                                               dir.x, dir.y, dir.z,
        //                                                               cur_id_geom, next_geometry_volume);
#ifdef DEBUG
        printf(" PE\n");
#endif
    }

    if (next_discrete_process == PHOTON_RAYLEIGH) {
        Rayleigh_SampleSecondaries_Livermore(particles, materials, photon_CS_table, E_index, id_mat, part_id);
        //printf("Rayleigh\n");
    }


    if (next_discrete_process == GEOMETRY_BOUNDARY) {
        // Debug
        //printf("id %i - pos %f %f %f - dir %f %f %f - Bnd - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
        //                                                                 dir.x, dir.y, dir.z,
        //                                                                 cur_id_geom, next_geometry_volume);
#ifdef DEBUG
        printf(" Geom\n");
#endif
    }

    //// Get discrete energy lost

    // If e- is not tracking drop its energy
    if (electron.endsimu == PARTICLE_DEAD) {
        discrete_loss += electron.E;
    }
    // If gamma is absorbed drop its energy
    if (particles.endsimu[part_id] == PARTICLE_DEAD) {
        discrete_loss += particles.E[part_id];
    }

    //// Handle sensitive object and singles detection


    if (parameters.record_singles_flag &&
            get_geometry_is_sensitive(geometry, cur_id_geom) && discrete_loss > 0) {

        printf("ID %i Cur id %i flag %i Pos %f %f %f Eloss %f\n", part_id, cur_id_geom,
               get_geometry_is_sensitive(geometry, cur_id_geom), pos.x, pos.y, pos.z, discrete_loss);

        if (singles.nb_hits[part_id] == 0) {
            singles.px[part_id] = pos.x*discrete_loss;
            singles.py[part_id] = pos.y*discrete_loss;
            singles.pz[part_id] = pos.z*discrete_loss;
            singles.E[part_id] = discrete_loss;
            singles.nb_hits[part_id] += 1;
        } else {
            singles.px[part_id] += pos.x*discrete_loss;
            singles.py[part_id] += pos.y*discrete_loss;
            singles.pz[part_id] += pos.z*discrete_loss;
            singles.E[part_id] += discrete_loss;
            singles.nb_hits[part_id] += 1;
        }

    }

    //// This part is for debuging and vrml viewer

    // Record this step if required
    if (history.record_flag == ENABLED) {
        history.cpu_record_a_step(particles, part_id);
    }

/*
    // DEBUGING: phasespace
    if (next_geometry_volume == 0 && particles.endsimu[part_id] == PARTICLE_ALIVE) {
        printf("%e %e %e %e %e %e %e\n", particles.E[part_id], pos.x, pos.y, pos.z, dir.x, dir.y, dir.z);
        particles.endsimu[part_id] = PARTICLE_DEAD;
        return;
    }
*/



}













#endif
