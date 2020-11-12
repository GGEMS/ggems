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
//#include "GGEMS/navigators/GGEMSPhotonNavigator.hh"

/*!
  \fn kernel void track_through_ggems_voxelized_solid(GGlong const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, global GGEMSVoxelizedSolidData* voxelized_solid_data, global GGshort const* label_data, global GGEMSParticleCrossSections* particle_cross_sections, global GGEMSMaterialTables* materials)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param random - pointer on random numbers
  \param voxelized_solid_data - pointer to voxelized solid data
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param materials - pointer on material in navigator
  \brief OpenCL kernel tracking particles within voxelized solid
  \return no returned value
*/
kernel void track_through_ggems_voxelized_solid(
  GGlong const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  global GGEMSVoxelizedSolidData const* voxelized_solid_data,
  global GGshort const* label_data,
  global GGEMSParticleCrossSections const* particle_cross_sections,
  global GGEMSMaterialTables const* materials)
{
  // Getting index of thread
  GGint global_id = get_global_id(0);

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
  GGfloat border_min_x = voxelized_solid_data->obb_geometry_.border_min_xyz_[0];
  GGfloat border_min_y = voxelized_solid_data->obb_geometry_.border_min_xyz_[1];
  GGfloat border_min_z = voxelized_solid_data->obb_geometry_.border_min_xyz_[2];

  // Get voxel size of voxelized solid
  GGfloat voxel_size_x = voxelized_solid_data->voxel_sizes_xyz_[0];
  GGfloat voxel_size_y = voxelized_solid_data->voxel_sizes_xyz_[1];
  GGfloat voxel_size_z = voxelized_solid_data->voxel_sizes_xyz_[2];

  // Get number of voxel of voxelized solid
  GGint number_of_voxels_x = voxelized_solid_data->number_of_voxels_xyz_[0];
  GGint number_of_voxels_y = voxelized_solid_data->number_of_voxels_xyz_[1];
  GGint number_of_voxels_z = voxelized_solid_data->number_of_voxels_xyz_[2];

  GGint4 index_voxel = {0, 0, 0, 0};
  GGshort material_id = 0;

  // Track particle until out of solid
  do {
    // Get index of voxelized phantom, x, y, z and w
    index_voxel.x = (GGint)((local_position.x - border_min_x) / voxel_size_x);
    index_voxel.y = (GGint)((local_position.y - border_min_y) / voxel_size_y);
    index_voxel.z = (GGint)((local_position.z - border_min_z) / voxel_size_z);
    index_voxel.w = index_voxel.x + index_voxel.y * voxel_size_x + index_voxel.z * voxel_size_x * voxel_size_y;

    // Get the material that compose this volume
    material_id = label_data[index_voxel.w];

    #ifdef GGEMS_TRACKING
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Particle id: %d\n", global_id);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Particle type: ");
      if (primary_particle->pname_[global_id] == PHOTON) printf("gamma\n");
      else if (primary_particle->pname_[global_id] == ELECTRON) printf("e-\n");
      else if (primary_particle->pname_[global_id] == POSITRON) printf("e+\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Position (x, y, z): %e %e %e mm\n", position.x/mm, position.y/mm, position.z/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Direction (x, y, z): %e %e %e\n", direction.x, direction.y, direction.z);
  //    printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Energy: %e keV\n", primary_particle->E_[global_id]/keV);
//      printf("\n");
//      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Solid id: %u\n", voxelized_solid_data->solid_id_);
//      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Nb voxels: %u %u %u\n", number_of_voxels_x, number_of_voxels_y, number_of_voxels_z);
//      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel size: %e %e %e mm\n", voxel_size_x/mm, voxel_size_y/mm, voxel_size_z/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Navigator X Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.x/mm, voxelized_solid_data->border_max_xyz_.x/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Navigator Y Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.y/mm, voxelized_solid_data->border_max_xyz_.y/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Navigator Z Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.z/mm, voxelized_solid_data->border_max_xyz_.z/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel X Borders: %e %e mm\n", kXMinVoxel/mm, kXMaxVoxel/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel Y Borders: %e %e mm\n", kYMinVoxel/mm, kYMaxVoxel/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel Z Borders: %e %e mm\n", kZMinVoxel/mm, kZMaxVoxel/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Index of current voxel (x, y, z): %d %d %d\n", index_voxel.x, index_voxel.y, index_voxel.z);
      //printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Global Index of current voxel: %d\n", index_voxel.w);
     // printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Material in voxel: %s\n", particle_cross_sections->material_names_[material_id]);
  //     printf("\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Next process: ");
  //     if (next_discrete_process == COMPTON_SCATTERING) printf("COMPTON_SCATTERING\n");
  //     if (next_discrete_process == PHOTOELECTRIC_EFFECT) printf("PHOTOELECTRIC_EFFECT\n");
  //     if (next_discrete_process == RAYLEIGH_SCATTERING) printf("RAYLEIGH_SCATTERING\n");
  //     if (next_discrete_process == TRANSPORTATION) printf("TRANSPORTATION\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Next interaction distance: %e mm\n", next_interaction_distance/mm);
    }
    #endif

    primary_particle->status_[global_id] = DEAD;
  } while (primary_particle->status_[global_id] == ALIVE);
  //while (primary_particle->status_[global_id] == ALIVE) {
    // Position of particle
    // GGfloat3 position = {
    //   primary_particle->px_[global_id],
    //   primary_particle->py_[global_id],
    //   primary_particle->pz_[global_id]
    // };

    // // Direction of particle
    // GGfloat3 direction = {
    //   primary_particle->dx_[global_id],
    //   primary_particle->dy_[global_id],
    //   primary_particle->dz_[global_id]
    // };

  //   // Get index of voxelized phantom, x, y, z and w (global index)
  //   GGint4 kIndexVoxel = {0, 0, 0, 0};
  //   kIndexVoxel.x = (GGint)((position.x + voxelized_solid_data->position_xyz_.x) / voxelized_solid_data->voxel_sizes_xyz_.x);
  //   kIndexVoxel.y = (GGint)((position.y + voxelized_solid_data->position_xyz_.y) / voxelized_solid_data->voxel_sizes_xyz_.y);
  //   kIndexVoxel.z = (GGint)((position.z + voxelized_solid_data->position_xyz_.z) / voxelized_solid_data->voxel_sizes_xyz_.z);
  //   kIndexVoxel.w = kIndexVoxel.x
  //     + kIndexVoxel.y * voxelized_solid_data->number_of_voxels_xyz_.x
  //     + kIndexVoxel.z * voxelized_solid_data->number_of_voxels_xyz_.x * voxelized_solid_data->number_of_voxels_xyz_.y;

  //   // Get the material that compose this volume
  //   GGuchar const kMaterialID = label_data[kIndexVoxel.w];

  //   // Find next discrete photon interaction
  //   GetPhotonNextInteraction(primary_particle, random, particle_cross_sections, kMaterialID, kParticleID);
  //   GGfloat next_interaction_distance = primary_particle->next_interaction_distance_[kParticleID];
  //   GGuchar next_discrete_process = primary_particle->next_discrete_process_[kParticleID];

  //   // Get the borders of the current voxel
  //   GGfloat const kXMinVoxel = kIndexVoxel.x*voxelized_solid_data->voxel_sizes_xyz_.x - voxelized_solid_data->position_xyz_.x;
  //   GGfloat const kYMinVoxel = kIndexVoxel.y*voxelized_solid_data->voxel_sizes_xyz_.y - voxelized_solid_data->position_xyz_.y;
  //   GGfloat const kZMinVoxel = kIndexVoxel.z*voxelized_solid_data->voxel_sizes_xyz_.z - voxelized_solid_data->position_xyz_.z;
  //   GGfloat const kXMaxVoxel = kXMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.x;
  //   GGfloat const kYMaxVoxel = kYMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.y;
  //   GGfloat const kZMaxVoxel = kZMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.z;

  //   // Get safety position of particle to be sure particle is inside voxel
  //   TransportGetSafetyInsideAABB(&position, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, GEOMETRY_TOLERANCE);

  //   // Get the distance to next boundary
  //   GGfloat const distance_to_next_boundary = ComputeDistanceToAABB(&position, &direction, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, GEOMETRY_TOLERANCE);

  //   // If distance to next boundary is inferior to distance to next interaction
  //   // we move particle to boundary
  //   if (distance_to_next_boundary <= next_interaction_distance) {
  //     next_interaction_distance = distance_to_next_boundary + GEOMETRY_TOLERANCE;
  //     next_discrete_process = TRANSPORTATION;
  //   }

  //   #ifdef GGEMS_TRACKING
  //   if (kParticleID == primary_particle->particle_tracking_id) {
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] ################################################################################\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Particle id: %d\n", kParticleID);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Particle type: ");
  //     if (primary_particle->pname_[kParticleID] == PHOTON) printf("gamma\n");
  //     else if (primary_particle->pname_[kParticleID] == ELECTRON) printf("e-\n");
  //     else if (primary_particle->pname_[kParticleID] == POSITRON) printf("e+\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Position (x, y, z): %e %e %e mm\n", position.x/mm, position.y/mm, position.z/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Direction (x, y, z): %e %e %e\n", direction.x, direction.y, direction.z);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Energy: %e keV\n", primary_particle->E_[kParticleID]/keV);
  //     printf("\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Navigator id: %u\n", voxelized_solid_data->navigator_id_);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Nb voxels: %u %u %u\n", voxelized_solid_data->number_of_voxels_xyz_.x, voxelized_solid_data->number_of_voxels_xyz_.y, voxelized_solid_data->number_of_voxels_xyz_.z);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel size: %e %e %e mm\n", voxelized_solid_data->voxel_sizes_xyz_.x/mm, voxelized_solid_data->voxel_sizes_xyz_.y/mm, voxelized_solid_data->voxel_sizes_xyz_.z/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Navigator X Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.x/mm, voxelized_solid_data->border_max_xyz_.x/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Navigator Y Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.y/mm, voxelized_solid_data->border_max_xyz_.y/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Navigator Z Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.z/mm, voxelized_solid_data->border_max_xyz_.z/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel X Borders: %e %e mm\n", kXMinVoxel/mm, kXMaxVoxel/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel Y Borders: %e %e mm\n", kYMinVoxel/mm, kYMaxVoxel/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Voxel Z Borders: %e %e mm\n", kZMinVoxel/mm, kZMaxVoxel/mm);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Index of current voxel (x, y, z): %d %d %d\n", kIndexVoxel.x, kIndexVoxel.y, kIndexVoxel.z);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Global Index of current voxel: %d\n", kIndexVoxel.w);
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Material in voxel: %s\n", particle_cross_sections->material_names_[kMaterialID]);
  //     printf("\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Next process: ");
  //     if (next_discrete_process == COMPTON_SCATTERING) printf("COMPTON_SCATTERING\n");
  //     if (next_discrete_process == PHOTOELECTRIC_EFFECT) printf("PHOTOELECTRIC_EFFECT\n");
  //     if (next_discrete_process == RAYLEIGH_SCATTERING) printf("RAYLEIGH_SCATTERING\n");
  //     if (next_discrete_process == TRANSPORTATION) printf("TRANSPORTATION\n");
  //     printf("[GGEMS OpenCL kernel track_through_ggems_voxelized_solid] Next interaction distance: %e mm\n", next_interaction_distance/mm);
  //   }
  //   #endif

  //   // Moving particle to next postion
  //   position = GGfloat3Add(position, GGfloat3Scale(direction, next_interaction_distance));

  //   // Get safety position of particle to be sure particle is outside voxel
  //   TransportGetSafetyOutsideAABB(&position, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, GEOMETRY_TOLERANCE);

  //   // Update TOF, true for photon only
  //   primary_particle->tof_[kParticleID] += next_interaction_distance * C_LIGHT;

  //   // Storing new position
  //   primary_particle->px_[kParticleID] = position.x;
  //   primary_particle->py_[kParticleID] = position.y;
  //   primary_particle->pz_[kParticleID] = position.z;

  //   // Checking if particle outside voxelized solid navigator
  //   if (!IsParticleInVoxelizedNavigator(&position, voxelized_solid_data)) {
  //     primary_particle->particle_navigator_distance_[kParticleID] = OUT_OF_WORLD; // Reset to initiale value
  //     primary_particle->navigator_id_[kParticleID] = 255; // Out of world navigator
  //     break;
  //   }

  //   // Resolve process if different of TRANSPORTATION
  //   if (next_discrete_process != TRANSPORTATION) {
  //     PhotonDiscreteProcess(primary_particle, random, materials, particle_cross_sections, kMaterialID, kParticleID);
  //   }
  //   primary_particle->status_[global_id] = DEAD;
  // }
}
