// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file TrackThroughGGEMSVoxelizedSolid.cl

  \brief OpenCL kernel tracking particles within voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday June 16, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"

#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"

#include "GGEMS/materials/GGEMSMaterialTables.hh"

#include "GGEMS/physics/GGEMSParticleCrossSections.hh"

#include "GGEMS/randoms/GGEMSRandom.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/navigators/GGEMSPhotonNavigator.hh"

#ifdef DOSIMETRY
#include "GGEMS/navigators/GGEMSDoseRecording.hh"
#endif

/*!
  \fn kernel void track_through_ggems_voxelized_solid(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, global GGEMSVoxelizedSolidData const* voxelized_solid_data, global GGshort const* label_data, global GGEMSParticleCrossSections const* particle_cross_sections, global GGEMSMaterialTables const* materials, GGfloat const threshold)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param random - pointer on random numbers
  \param voxelized_solid_data - pointer to voxelized solid data
  \param label_data - pointer storing label of material
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param materials - pointer on material in navigator
  \param threshold - energy threshold
  \brief OpenCL kernel tracking particles within voxelized solid
*/
kernel void track_through_ggems_voxelized_solid(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  global GGEMSVoxelizedSolidData const* voxelized_solid_data,
  global GGshort const* label_data,
  global GGEMSParticleCrossSections const* particle_cross_sections,
  global GGEMSMaterialTables const* materials,
  GGfloat const threshold
  #ifdef DOSIMETRY
  ,global GGEMSDoseParams* dose_params,
  global GGDosiType* edep_tracking,
  global GGDosiType* edep_squared_tracking,
  global GGint* hit_tracking,
  global GGint* photon_tracking
  #endif
)
{
  // Getting index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // Checking if the current navigator is the selected navigator
  if (primary_particle->solid_id_[global_id] != voxelized_solid_data->solid_id_) return;

  // Checking status of particle
  if (primary_particle->status_[global_id] == DEAD) {
    #ifdef GGEMS_TRACKING
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] The particle id %d is dead!!!\n", global_id);
    }
    #endif
    return;
  }

  // Get position and direction in OBB coordinate (local)
  GGfloat44 tmp_matrix_transformation = {
    {voxelized_solid_data->obb_geometry_.matrix_transformation_.m0_[0], voxelized_solid_data->obb_geometry_.matrix_transformation_.m0_[1], voxelized_solid_data->obb_geometry_.matrix_transformation_.m0_[2], voxelized_solid_data->obb_geometry_.matrix_transformation_.m0_[3]},
    {voxelized_solid_data->obb_geometry_.matrix_transformation_.m1_[0], voxelized_solid_data->obb_geometry_.matrix_transformation_.m1_[1], voxelized_solid_data->obb_geometry_.matrix_transformation_.m1_[2], voxelized_solid_data->obb_geometry_.matrix_transformation_.m1_[3]},
    {voxelized_solid_data->obb_geometry_.matrix_transformation_.m2_[0], voxelized_solid_data->obb_geometry_.matrix_transformation_.m2_[1], voxelized_solid_data->obb_geometry_.matrix_transformation_.m2_[2], voxelized_solid_data->obb_geometry_.matrix_transformation_.m2_[3]},
    {voxelized_solid_data->obb_geometry_.matrix_transformation_.m3_[0], voxelized_solid_data->obb_geometry_.matrix_transformation_.m3_[1], voxelized_solid_data->obb_geometry_.matrix_transformation_.m3_[2], voxelized_solid_data->obb_geometry_.matrix_transformation_.m3_[3]}
  };

  // Get the position and direction in local OBB coordinate
  GGfloat3 global_position = {primary_particle->px_[global_id], primary_particle->py_[global_id], primary_particle->pz_[global_id]};
  GGfloat3 global_direction = {primary_particle->dx_[global_id], primary_particle->dy_[global_id], primary_particle->dz_[global_id]};
  GGfloat3 local_position = GlobalToLocalPosition(&tmp_matrix_transformation, &global_position);
  GGfloat3 local_direction = GlobalToLocalDirection(&tmp_matrix_transformation, &global_direction);

  // Get borders of OBB
  GGfloat3 border_min = voxelized_solid_data->obb_geometry_.border_min_xyz_;
  GGfloat3 border_max = voxelized_solid_data->obb_geometry_.border_max_xyz_;

  GGfloat3 voxel_size = voxelized_solid_data->voxel_sizes_xyz_;
  GGint3 number_of_voxels = voxelized_solid_data->number_of_voxels_xyz_;

  // Track particle until out of solid
  do {
    // Get index of voxelized phantom, x, y, z
    GGint3 voxel_id = convert_int3((local_position - border_min) / voxel_size);

    // Get the material that compose this volume
    GGshort material_id = label_data[voxel_id.x + voxel_id.y * number_of_voxels.x + voxel_id.z * number_of_voxels.x * number_of_voxels.y];

    // Find next discrete photon interaction
    GetPhotonNextInteraction(primary_particle, random, particle_cross_sections, material_id, global_id);
    GGfloat next_interaction_distance = primary_particle->next_interaction_distance_[global_id];
    GGchar next_discrete_process = primary_particle->next_discrete_process_[global_id];

    // Get the borders of the current voxel
    GGfloat3 voxel_border_min = border_min +  convert_float3(voxel_id)*voxel_size;
    GGfloat3 voxel_border_max = voxel_border_min + voxel_size;

    // Get safety position of particle to be sure particle is inside voxel
    TransportGetSafetyInsideAABB(
      &local_position,
      voxel_border_min.x, voxel_border_max.x,
      voxel_border_min.y, voxel_border_max.y,
      voxel_border_min.z, voxel_border_max.z,
      GEOMETRY_TOLERANCE
    );

    // Get the distance to next boundary
    GGfloat distance_to_next_boundary = ComputeDistanceToAABB(
      &local_position, &local_direction,
      voxel_border_min.x, voxel_border_max.x,
      voxel_border_min.y, voxel_border_max.y,
      voxel_border_min.z, voxel_border_max.z,
      GEOMETRY_TOLERANCE
    );

    // If distance to next boundary is inferior to distance to next interaction we move particle to boundary
    if (distance_to_next_boundary <= next_interaction_distance) {
      next_interaction_distance = distance_to_next_boundary + GEOMETRY_TOLERANCE;
      next_discrete_process = TRANSPORTATION;
      #ifdef DOSIMETRY
      if (photon_tracking) dose_photon_tracking(dose_params, photon_tracking, &local_position);
      #endif
    }

    #ifdef GGEMS_TRACKING
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Particle id: %d\n", global_id);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Particle type: ");
      if (primary_particle->pname_[global_id] == PHOTON) printf("gamma\n");
      else if (primary_particle->pname_[global_id] == ELECTRON) printf("e-\n");
      else if (primary_particle->pname_[global_id] == POSITRON) printf("e+\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Local position (x, y, z): %e %e %e mm\n", local_position.x/mm, local_position.y/mm, local_position.z/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Local direction (x, y, z): %e %e %e\n", local_direction.x, local_direction.y, local_direction.z);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Energy: %e keV\n", primary_particle->E_[global_id]/keV);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Solid id: %u\n", voxelized_solid_data->solid_id_);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Nb voxels: %u %u %u\n", number_of_voxels.x, number_of_voxels.y, number_of_voxels.z);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel size: %e %e %e mm\n", voxel_size.x/mm, voxel_size.y/mm, voxel_size.z/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Solid X Borders: %e %e mm\n", border_min.x/mm, border_max.x/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Solid Y Borders: %e %e mm\n", border_min.y/mm, border_max.y/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Solid Z Borders: %e %e mm\n", border_min.z/mm, border_max.z/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel X Borders: %e %e mm\n", voxel_border_min.x/mm, voxel_border_max.x/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel Y Borders: %e %e mm\n", voxel_border_min.y/mm, voxel_border_max.y/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel Z Borders: %e %e mm\n", voxel_border_min.z/mm, voxel_border_max.z/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Index of current voxel (x, y, z): %d %d %d\n", voxel_id.x, voxel_id.y, voxel_id.z);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Material in voxel: %s\n", particle_cross_sections->material_names_[material_id]);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Next process: ");
      if (next_discrete_process == COMPTON_SCATTERING) printf("COMPTON_SCATTERING\n");
      if (next_discrete_process == PHOTOELECTRIC_EFFECT) printf("PHOTOELECTRIC_EFFECT\n");
      if (next_discrete_process == RAYLEIGH_SCATTERING) printf("RAYLEIGH_SCATTERING\n");
      if (next_discrete_process == TRANSPORTATION) printf("TRANSPORTATION\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Next interaction distance: %e mm\n", next_interaction_distance/mm);
    }
    #endif

    // Moving particle to next position
    local_position = local_position + local_direction*next_interaction_distance;

    // Get safety position of particle to be sure particle is outside voxel
    TransportGetSafetyOutsideAABB(
      &local_position,
      voxel_border_min.x, voxel_border_max.x,
      voxel_border_min.y, voxel_border_max.y,
      voxel_border_min.z, voxel_border_max.z,
      GEOMETRY_TOLERANCE
    );

    //  Checking if particle outside solid, still in local
    if (!IsParticleInAABB(&local_position, border_min.x, border_max.x, border_min.y, border_max.y, border_min.z, border_max.z, GEOMETRY_TOLERANCE)) {
      primary_particle->particle_solid_distance_[global_id] = OUT_OF_WORLD; // Reset to initiale value
      primary_particle->solid_id_[global_id] = -1; // Out of world
      break;
    }

    // Storing new position in local
    primary_particle->px_[global_id] = local_position.x;
    primary_particle->py_[global_id] = local_position.y;
    primary_particle->pz_[global_id] = local_position.z;

    // Resolve process if different of TRANSPORTATION
    if (next_discrete_process != TRANSPORTATION) {
      #ifdef DOSIMETRY
      GGfloat edep = primary_particle->E_[global_id];
      #endif

      PhotonDiscreteProcess(primary_particle, random, materials, particle_cross_sections, material_id, global_id);

      #ifdef DOSIMETRY
      edep -= primary_particle->E_[global_id];
      dose_record_standard(dose_params, edep_tracking, edep_squared_tracking, hit_tracking, edep, &local_position);
      #endif

      local_direction.x = primary_particle->dx_[global_id];
      local_direction.y = primary_particle->dy_[global_id];
      local_direction.z = primary_particle->dz_[global_id];
    }

    // Apply threshold
    if (primary_particle->E_[global_id] <= materials->photon_energy_cut_[material_id]) {
      #ifdef DOSIMETRY
      dose_record_standard(dose_params, edep_tracking, edep_squared_tracking, hit_tracking, primary_particle->E_[global_id], &local_position);
      #endif
      primary_particle->status_[global_id] = DEAD;
    }
  } while (primary_particle->status_[global_id] == ALIVE);

  // Convert to global position
  global_position = LocalToGlobalPosition(&tmp_matrix_transformation, &local_position);
  primary_particle->px_[global_id] = global_position.x;
  primary_particle->py_[global_id] = global_position.y;
  primary_particle->pz_[global_id] = global_position.z;

  // Convert to global direction
  global_direction = LocalToGlobalDirection(&tmp_matrix_transformation, &local_direction);
  primary_particle->dx_[global_id] = global_direction.x;
  primary_particle->dy_[global_id] = global_direction.y;
  primary_particle->dz_[global_id] = global_direction.z;
}
