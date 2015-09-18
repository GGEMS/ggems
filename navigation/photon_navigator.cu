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
__host__ __device__ void photon_navigator(ParticleStack &particles, ui32 part_id,
                          Scene geometry, MaterialsTable materials,
                          PhotonCrossSectionTable photon_CS_table,
                          GlobalSimulationParameters parameters,
                          Pulses &pulses) {

  
  
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
    
    ui32 adr_geom = geometry.ptr_objects[cur_id_geom];
    ui32 obj_type = (ui32)geometry.data_objects[adr_geom+ADR_OBJ_TYPE];
      
   // printf(" obj type %d \n", obj_type);
    
    
  /*  if (part_id == 6614629 ) {
      
      
       if (test_point_AABB(pos, (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN], (f64)geometry.data_objects[adr_geom+ADR_AABB_XMAX],
              (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN], (f64)geometry.data_objects[adr_geom+ADR_AABB_YMAX], 
              (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN], (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMAX])) {
        
          printf("INSIDE \n");
         
       } else {
         printf("NOT .... \n");
       }
      
        if (cur_id_geom==7) { 
    
          f64xyz posinvox;
        posinvox.x = pos.x - (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN]; // -= xmin
        posinvox.y = pos.y - (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN]; // -= ymin
        posinvox.z = pos.z - (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; // -= zmin
        printf("posinvox %f %f %f \n", posinvox.x, posinvox.y, posinvox.z);
        // Get spacing
        f64xyz s;
        s.x = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SX];
        s.y = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SY];
        s.z = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SZ];
        // Get the voxel index
        ui32xyz ind;
        ind.x = (ui32)(posinvox.x / s.x);
        ind.y = (ui32)(posinvox.y / s.y);
        ind.z = (ui32)(posinvox.z / s.z);

        printf("Ind %i %i %i\n", ind.x, ind.y, ind.z);
          
        f64 xmin, ymin, xmax, ymax, zmin, zmax;
        xmin = ind.x*s.x + (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN]; xmax = xmin+s.x;
        ymin = ind.y*s.y + (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN]; ymax = ymin+s.y;
        zmin = ind.z*s.z + (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; zmax = zmin+s.z;
    
          
        }
    }*/
       
    // If the particle hits the SPECThead, determine in which layer it is
    if (obj_type == SPECTHEAD) {   
        
        //Check all SPECThead children
        cur_id_geom = get_current_geometry_volume(geometry, cur_id_geom, pos);
   
        // Update the object type
        adr_geom = geometry.ptr_objects[cur_id_geom];
        obj_type = (ui32)geometry.data_objects[adr_geom+ADR_OBJ_TYPE];

    } 
    
    // Check if particle is really inside the voxelized phantom 
 /*   if (obj_type == VOXELIZED) {     
         if (!test_point_AABB(pos, (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN], (f64)geometry.data_objects[adr_geom+ADR_AABB_XMAX],
              (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN], (f64)geometry.data_objects[adr_geom+ADR_AABB_YMAX], 
              (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN], (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMAX])) {

              f64xyz posinvox;
              posinvox.x = pos.x - (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN]; // -= xmin
              posinvox.y = pos.y - (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN]; // -= ymin
              posinvox.z = pos.z - (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; // -= zmin
              printf("posinvox %f %f %f \n", posinvox.x, posinvox.y, posinvox.z);
              // Get spacing
              f64xyz s;
              s.x = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SX];
              s.y = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SY];
              s.z = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SZ];
              // Get the voxel index
              ui32xyz ind;
              ind.x = (ui32)(posinvox.x / s.x);
              ind.y = (ui32)(posinvox.y / s.y);
              ind.z = (ui32)(posinvox.z / s.z);

              printf("Ind %i %i %i\n", ind.x, ind.y, ind.z);
           
              //cur_id_geom = geometry.mother_node[cur_id_geom]; 
              printf("WARNING Particle %d outside voxelized phantom id_geom %d \n", part_id, cur_id_geom);
         }
    }*/
    
    // Get the material that compose this volume
    ui32 id_mat = get_geometry_material(geometry, cur_id_geom, pos);
 
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

        if (interaction_distance < next_interaction_distance) {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_PHOTOELECTRIC;
          //  printf("PHOTOELEC %f ", next_interaction_distance);
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

    // printf("Before geom\n");

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
    

    
    /*if (part_id == 7050335 || part_id == 8816730 ) {
      
        printf("id %d mat %d process %d dist %f cur_geom %d next_geom %d \n", part_id, id_mat, next_discrete_process, next_interaction_distance, cur_id_geom, next_geometry_volume);

    }*/
   // printf("move particle %f %f %f dir %f %f %f \n", pos.x, pos.y, pos.z, dir.x, dir.y, dir.z);
     
    // Move the particle
    pos = fxyz_add(pos, fxyz_scale(dir, next_interaction_distance));
    
    

    // Update TOF
    particles.tof[part_id] += c_light * next_interaction_distance;

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
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    f32 discrete_loss = 0.0f;
    SecParticle electron;
    electron.E = 0;

  
    
    if (next_discrete_process == PHOTON_COMPTON) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                 cutE
        electron = Compton_SampleSecondaries_standard(particles, 0.0, part_id, parameters);
    }

    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {

        //   TODO: cutE = materials.electron_cut_energy[mat]                                               cutE
        electron = Photoelec_SampleSecondaries_standard(particles, materials, photon_CS_table,
                                                                    E_index, 0.0, id_mat, part_id, parameters);
    }

    if (next_discrete_process == PHOTON_RAYLEIGH) {
        Rayleigh_SampleSecondaries_Livermore(particles, materials, photon_CS_table, E_index, id_mat, part_id);

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

    if (parameters.digitizer_flag &&
            get_geometry_is_sensitive(geometry, cur_id_geom) && discrete_loss > 0) {
      
        ui32 adr_geom = geometry.ptr_objects[cur_id_geom];
      
        f64xyz obb_center;
        obb_center.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_X];
        obb_center.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Y];
        obb_center.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Z];
       
        f64xyz u, v, w;
        u.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UX];
        u.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UY];
        u.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UZ];
        v.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VX];
        v.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VY];
        v.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VZ];
        w.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WX];
        w.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WY];
        w.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WZ];
      
        f64xyz pos_obb = fxyz_sub(pos, obb_center);
        f64xyz pos_aabb;
        pos_aabb.x = fxyz_dot(pos_obb, u);
        pos_aabb.y = fxyz_dot(pos_obb, v);
        pos_aabb.z = fxyz_dot(pos_obb, w);
       
        // First hit - first pulse
        if (pulses.pu1_nb_hits[part_id] == 0) {
            pulses.pu1_px[part_id] = pos_aabb.x*discrete_loss;
            pulses.pu1_py[part_id] = pos_aabb.y*discrete_loss;
            pulses.pu1_pz[part_id] = pos_aabb.z*discrete_loss;
            pulses.pu1_E[part_id] = discrete_loss;
            pulses.pu1_tof[part_id] = particles.tof[part_id]; // Time is defined for the first hit
            pulses.pu1_nb_hits[part_id] += 1;
            pulses.pu1_id_geom[part_id] = cur_id_geom;

            
        } else {

            // Others hits - first pulse
            if (cur_id_geom == pulses.pu1_id_geom[part_id]) {
                pulses.pu1_px[part_id] += pos_aabb.x*discrete_loss;
                pulses.pu1_py[part_id] += pos_aabb.y*discrete_loss;
                pulses.pu1_pz[part_id] += pos_aabb.z*discrete_loss;
                pulses.pu1_E[part_id] += discrete_loss;
                pulses.pu1_nb_hits[part_id] += 1;

            } else {

                // First hit - second pulse
                if (pulses.pu2_nb_hits[part_id] == 0) {
                    pulses.pu2_px[part_id] = pos_aabb.x*discrete_loss;
                    pulses.pu2_py[part_id] = pos_aabb.y*discrete_loss;
                    pulses.pu2_pz[part_id] = pos_aabb.z*discrete_loss;
                    pulses.pu2_E[part_id] = discrete_loss;
                    pulses.pu2_tof[part_id] = particles.tof[part_id]; // Time is defined for the first hit
                    pulses.pu2_nb_hits[part_id] += 1;
                    pulses.pu2_id_geom[part_id] = cur_id_geom;

                } else {
                    // Others hits - second pulse
                    pulses.pu2_px[part_id] += pos_aabb.x*discrete_loss;
                    pulses.pu2_py[part_id] += pos_aabb.y*discrete_loss;
                    pulses.pu2_pz[part_id] += pos_aabb.z*discrete_loss;
                    pulses.pu2_E[part_id] += discrete_loss;
                    pulses.pu2_nb_hits[part_id] += 1;
                }
            }
        }

    } // Digitizer
}


//// SPECIAL CASE RAYTRACING SPECT
__host__ __device__ void photon_navigator_raytracing_colli(ParticleStack &particles, ui32 part_id,
                          Scene geometry, MaterialsTable materials,
                          PhotonCrossSectionTable photon_CS_table,
                          GlobalSimulationParameters parameters,
                          Pulses &pulses) {

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
    
    f64 next_interaction_distance = F64_MAX;
    f64 interaction_distance;
        
    // Get the current volume containing the particle
    ui32 cur_id_geom = particles.geometry_id[part_id];

    ui32 adr_geom = geometry.ptr_objects[cur_id_geom];
    ui32 obj_type = (ui32)geometry.data_objects[adr_geom+ADR_OBJ_TYPE];
      
    ui32 next_geometry_volume = cur_id_geom;
    
    // If the particle hits the SPECThead, determine in which layer it is
    if (obj_type == SPECTHEAD) {            
        //Check all SPECThead children
        cur_id_geom = get_current_geometry_volume(geometry, cur_id_geom, pos);
        // Update the object type
        adr_geom = geometry.ptr_objects[cur_id_geom];
        obj_type = (ui32)geometry.data_objects[adr_geom+ADR_OBJ_TYPE];
    } 
    
    if (obj_type == COLLI) {
        
        f64 aabb_xmin = (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN];
        f64 aabb_xmax = (f64)geometry.data_objects[adr_geom+ADR_AABB_XMAX];
        f64 aabb_ymin = (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN];
        f64 aabb_ymax = (f64)geometry.data_objects[adr_geom+ADR_AABB_YMAX];
        f64 aabb_zmin = (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN];
        f64 aabb_zmax = (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMAX];
        
        f64xyz colli_center;
        colli_center.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_X];
        colli_center.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Y];
        colli_center.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Z];
        
        f64xyz u, v, w;
        u.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UX];
        u.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UY];
        u.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UZ];
        v.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VX];
        v.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VY];
        v.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VZ];
        w.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WX];
        w.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WY];
        w.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WZ];
        
        i32 hex = GetHexIndex(pos, geometry, adr_geom, colli_center, u, v, w);
        
        if (hex < 0) {
            particles.endsimu[part_id] = PARTICLE_DEAD;
        }
        else {
            
            f64 half_colli_size_x = (aabb_xmax - aabb_xmin) * 0.5;
            f64 half_colli_size_y = (aabb_ymax - aabb_ymin) * 0.5;
            f64 half_colli_size_z = (aabb_zmax - aabb_zmin) * 0.5;
            
            f64 hole_radius = (f64)geometry.data_objects[adr_geom+ADR_COLLI_HOLE_RADIUS];
            
            ui32 nb_hex = (i32)geometry.data_objects[adr_geom + ADR_COLLI_NB_HEXAGONS];
            ui32 ind_y = adr_geom + ADR_COLLI_CENTEROFHEXAGONS;
            ui32 ind_z = adr_geom + ADR_COLLI_CENTEROFHEXAGONS + nb_hex;
            
            f64xyz ray_obb = fxyz_sub(pos, colli_center);
            
            f64xyz p;
            p.x = fxyz_dot(ray_obb, u);
            p.y = fxyz_dot(ray_obb, v);
            p.z = fxyz_dot(ray_obb, w);
            f64xyz d;
            d.x = fxyz_dot(dir, u);
            d.y = fxyz_dot(dir, v);
            d.z = fxyz_dot(dir, w);
          
            f64xyz temp;
            temp.x = p.x;
            temp.y = p.y - (f64)geometry.data_objects[ind_y+hex];
            temp.z = p.z - (f64)geometry.data_objects[ind_z+hex];
                  
           // printf("centerofhex y %f z %f \n", geometry.data_objects[adr_geom+ADR_COLLI_CENTEROFHEXAGONS_Y+hex], 
                //   geometry.data_objects[adr_geom+ADR_COLLI_CENTEROFHEXAGONS_Z+hex] );
           
            f64 distance = hit_ray_septa(temp, d, half_colli_size_x, hole_radius, colli_center, u, v, w);
            
            // Move the particle
            f64xyz pos_temp = fxyz_add(pos, fxyz_scale(dir, distance));
            
            i32 hex_test = GetHexIndex(pos_temp, geometry, adr_geom, colli_center, u, v, w);
            
            if (hex_test == hex) {
              
                f64 next_interaction_distance = distance + EPSILON3; // Overshoot
                f64xyz pos_test = fxyz_add(pos, fxyz_scale(dir, next_interaction_distance));
                next_geometry_volume = geometry.mother_node[cur_id_geom];
                // Update TOF
                particles.tof[part_id] += c_light * next_interaction_distance;
                particles.px[part_id] = pos_test.x;
                particles.py[part_id] = pos_test.y;
                particles.pz[part_id] = pos_test.z;
              
            } else {
                particles.endsimu[part_id] = PARTICLE_DEAD;
            }
            // if particle is not outside colli box, kill the particle (hit a septa)
            /*if (test_point_OBB(pos_test, aabb_xmin, aabb_xmax, aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax, colli_center, u, v, w)) {
                particles.endsimu[part_id] = PARTICLE_DEAD;
            }
            else {
                next_geometry_volume = geometry.mother_node[cur_id_geom];
                // Update TOF
                particles.tof[part_id] += c_light * next_interaction_distance;
                particles.px[part_id] = pos_test.x;
                particles.py[part_id] = pos_test.y;
                particles.pz[part_id] = pos_test.z;
            }*/
        }
    }
    else {
    
        // Get the material that compose this volume
        ui32 id_mat = get_geometry_material(geometry, cur_id_geom, pos);

        //// Find next discrete interaction ///////////////////////////////////////

        ui8 next_discrete_process = 0;
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
        pos = fxyz_add(pos, fxyz_scale(dir, next_interaction_distance));

        // Update TOF
        particles.tof[part_id] += c_light * next_interaction_distance;

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
            return;
        }

        //// Apply discrete process //////////////////////////////////////////////////

        f32 discrete_loss = 0.0f;
        SecParticle electron;
        electron.E = 0;

        if (next_discrete_process == PHOTON_COMPTON) {

            //   TODO: cutE = materials.electron_cut_energy[mat]                 cutE
            electron = Compton_SampleSecondaries_standard(particles, 0.0, part_id, parameters);

        }

        if (next_discrete_process == PHOTON_PHOTOELECTRIC) {

            //   TODO: cutE = materials.electron_cut_energy[mat]                                               cutE
            electron = Photoelec_SampleSecondaries_standard(particles, materials, photon_CS_table,
                                                                        E_index, 0.0, id_mat, part_id, parameters);

        }

        if (next_discrete_process == PHOTON_RAYLEIGH) {
            Rayleigh_SampleSecondaries_Livermore(particles, materials, photon_CS_table, E_index, id_mat, part_id);
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

        if (parameters.digitizer_flag &&
                get_geometry_is_sensitive(geometry, cur_id_geom) && discrete_loss > 0) {
            
            ui32 adr_geom = geometry.ptr_objects[cur_id_geom];
      
            f64xyz obb_center;
            obb_center.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_X];
            obb_center.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Y];
            obb_center.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Z];
          
            f64xyz u, v, w;
            u.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UX];
            u.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UY];
            u.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UZ];
            v.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VX];
            v.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VY];
            v.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VZ];
            w.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WX];
            w.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WY];
            w.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WZ];
          
            f64xyz pos_obb = fxyz_sub(pos, obb_center);
            f64xyz pos_aabb;
            pos_aabb.x = fxyz_dot(pos_obb, u);
            pos_aabb.y = fxyz_dot(pos_obb, v);
            pos_aabb.z = fxyz_dot(pos_obb, w);
          
            // First hit - first pulse
            if (pulses.pu1_nb_hits[part_id] == 0) {
                pulses.pu1_px[part_id] = pos_aabb.x*discrete_loss;
                pulses.pu1_py[part_id] = pos_aabb.y*discrete_loss;
                pulses.pu1_pz[part_id] = pos_aabb.z*discrete_loss;
                pulses.pu1_E[part_id] = discrete_loss;
                pulses.pu1_tof[part_id] = particles.tof[part_id]; // Time is defined for the first hit
                pulses.pu1_nb_hits[part_id] += 1;
                pulses.pu1_id_geom[part_id] = cur_id_geom;
                //printf("pulse %d pos %f %f %f \n", part_id, pulses.pu1_px[part_id], 
                  //              pulses.pu1_py[part_id], pulses.pu1_pz[part_id]);

            } else {

                // Others hits - first pulse
                if (cur_id_geom == pulses.pu1_id_geom[part_id]) {
                    pulses.pu1_px[part_id] += pos_aabb.x*discrete_loss;
                    pulses.pu1_py[part_id] += pos_aabb.y*discrete_loss;
                    pulses.pu1_pz[part_id] += pos_aabb.z*discrete_loss;
                    pulses.pu1_E[part_id] += discrete_loss;
                    pulses.pu1_nb_hits[part_id] += 1;

                } else {

                    // First hit - second pulse
                    if (pulses.pu2_nb_hits[part_id] == 0) {
                        pulses.pu2_px[part_id] = pos_aabb.x*discrete_loss;
                        pulses.pu2_py[part_id] = pos_aabb.y*discrete_loss;
                        pulses.pu2_pz[part_id] = pos_aabb.z*discrete_loss;
                        pulses.pu2_E[part_id] = discrete_loss;
                        pulses.pu2_tof[part_id] = particles.tof[part_id]; // Time is defined for the first hit
                        pulses.pu2_nb_hits[part_id] += 1;
                        pulses.pu2_id_geom[part_id] = cur_id_geom;

                    } else {
                        // Others hits - second pulse
                        pulses.pu2_px[part_id] += pos_aabb.x*discrete_loss;
                        pulses.pu2_py[part_id] += pos_aabb.y*discrete_loss;
                        pulses.pu2_pz[part_id] += pos_aabb.z*discrete_loss;
                        pulses.pu2_E[part_id] += discrete_loss;
                        pulses.pu2_nb_hits[part_id] += 1;
                    }
                }
            }
        } // Digitizer
    }
}

#endif
