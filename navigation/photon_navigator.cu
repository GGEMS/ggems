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
                          ImageDetector &panel_detector,
                          HistoryBuilder &history) {


    // Read position
    f32xyz pos;
    pos.x = particles.px[part_id];
    pos.y = particles.py[part_id];
    pos.z = particles.pz[part_id];

    // Read direction
    f32xyz dir;
    dir.x = particles.dx[part_id];
    dir.y = particles.dy[part_id];
    dir.z = particles.dz[part_id];

    // Get the current volume containing the particle
    ui32 cur_id_geom = particles.geometry_id[part_id];

    // Get the material that compose this volume
    ui32 id_mat = get_geometry_material(geometry, cur_id_geom, pos);

    printf("     begin %i\n", part_id);
    printf("     Cur id geom %i mat %i\n", cur_id_geom, id_mat);
    printf("     InitPos %f %f %f\n", pos.x, pos.y, pos.z);

    //// Find next discrete interaction ///////////////////////////////////////

    f32 next_interaction_distance = FLT_MAX;
    unsigned char next_discrete_process = 0;
    ui32 next_geometry_volume = cur_id_geom;
    f32 interaction_distance;
    f32 cross_section;

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
        if (interaction_distance < next_interaction_distance) {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_RAYLEIGH;
        }

    }

    //// Get the next distance boundary volume /////////////////////////////////

    //printf("Before geom\n");

    ui32 hit_id_geom = 0;
    get_next_geometry_boundary(geometry, cur_id_geom, pos, dir, interaction_distance, hit_id_geom);
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
    pos = f3_add(pos, f3_scale(dir, next_interaction_distance));    

    // TODO
    //particles.tof[id] += gpu_speed_of_light * next_interaction_distance;

    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    particles.geometry_id[part_id] = next_geometry_volume;

    // Check world boundary
    f32 xmin = geometry.data_objects[ADR_AABB_XMIN]; // adr_world_geom = 0
    f32 xmax = geometry.data_objects[ADR_AABB_XMAX];
    f32 ymin = geometry.data_objects[ADR_AABB_YMIN];
    f32 ymax = geometry.data_objects[ADR_AABB_YMAX];
    f32 zmin = geometry.data_objects[ADR_AABB_ZMIN];
    f32 zmax = geometry.data_objects[ADR_AABB_ZMAX];

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

    //f32 discrete_loss = 0.0f;

    printf("     Dist %f NextVol %i pos %f %f %f ", next_interaction_distance, next_geometry_volume, pos.x, pos.y, pos.z);

    if (next_discrete_process == PHOTON_COMPTON) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                 cutE
        SecParticle electron = Compton_SampleSecondaries_standard(particles, 0.0, part_id, parameters);

        // Debug
        //printf("id %i - pos %f %f %f - dir %f %f %f - Cmpt - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
        //                                                                 dir.x, dir.y, dir.z,
        //                                                                 cur_id_geom, next_geometry_volume);
        printf(" Compton\n");
    }

    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                                               cutE
        SecParticle electron = Photoelec_SampleSecondaries_standard(particles, materials, photon_CS_table,
                                                                    E_index, 0.0, id_mat, part_id, parameters);


        // Debug
        //printf("id %i - pos %f %f %f - dir %f %f %f - PE - geom cur %i hit %i\n", part_id, pos.x, pos.y, pos.z,
        //                                                               dir.x, dir.y, dir.z,
        //                                                               cur_id_geom, next_geometry_volume);
        printf(" PE\n");
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
        printf(" Geom\n");
    }


    // WARNING: drop energy for every "dead" particle (gamma + e-)

    // Local deposition if this photon was absorbed
    //if (particles.endsimu[part_id] == PARTICLE_DEAD) discrete_loss = particles.E[part_id];

    //// Handle detector ////////////////////////////////////////
/*
    // If a detector is defined
    if (panel_detector.data != NULL) {
        if (next_geometry_volume == panel_detector.geometry_id) {

            // Change particle frame (into voxelized volume)
            pos.x -= panel_detector.xmin;
            pos.y -= panel_detector.ymin;
            pos.z -= panel_detector.zmin;
            // Get the voxel index
            int3 ind;
            ind.x = (ui32)(pos.x / panel_detector.sx);
            ind.y = (ui32)(pos.y / panel_detector.sy);
            ind.z = (ui32)(pos.z / panel_detector.sz);
            // Assertion
            assert(ind.x < panel_detector.nx);
            assert(ind.y < panel_detector.ny);
            assert(ind.z < panel_detector.nz);
            // Count a hit
            ui32 abs_ind = ind.z * (panel_detector.nx*panel_detector.ny) +
                                   ind.y * panel_detector.nx +
                                   ind.x;
            panel_detector.data[abs_ind] += 1.0f;
            panel_detector.countp++;

            //printf("Ct in %i\n", panel_detector.countp);

            // FIXME - Kill the particle for a simple simulation
            //         usually each hit have to be stored within the detector
            particles.endsimu[part_id] = PARTICLE_DEAD;

        }
    }
*/

/*
    // Record this step if required
    if (history.record_flag == ENABLED) {
        history.cpu_record_a_step(particles, part_id);
    }
*/

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
