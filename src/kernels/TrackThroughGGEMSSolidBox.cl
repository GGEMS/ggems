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
  \file TrackThroughGGEMSSolidBox.cl

  \brief OpenCL kernel tracking particles within solid box

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 25, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"

#include "GGEMS/geometries/GGEMSSolidBoxData.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"

#include "GGEMS/materials/GGEMSMaterialTables.hh"

#include "GGEMS/physics/GGEMSParticleCrossSections.hh"

#include "GGEMS/randoms/GGEMSRandom.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/navigators/GGEMSPhotonNavigator.hh"

/*!
  \fn kernel void track_through_ggems_solid_box(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, global GGEMSSolidBoxData const* solid_box_data, global GGuchar const* label_data, global GGEMSParticleCrossSections const* particle_cross_sections, global GGEMSMaterialTables const* materials, GGfloat const threshold, global GGint* histogram)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param random - pointer on random numbers
  \param solid_box_data - pointer to solid box data
  \param label_data - pointer storing label of material (empty buffer here, 1 material only)
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param materials - pointer on material in navigator
  \param threshold - energy threshold
  \param histogram - pointer to buffer storing histogram
  \brief OpenCL kernel tracking particles within voxelized solid
*/
kernel void track_through_ggems_solid_box(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  global GGEMSSolidBoxData const* solid_box_data,
  global GGuchar const* label_data,
  global GGEMSParticleCrossSections const* particle_cross_sections,
  global GGEMSMaterialTables const* materials,
  GGfloat const threshold
  #ifdef HISTOGRAM
  ,global GGint* histogram,
  global GGint* scatter_histogram
  #endif
)
{
  // Getting index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // Checking if the current navigator is the selected navigator
  if (primary_particle->solid_id_[global_id] != solid_box_data->solid_id_) return;

  // Checking status of particle
  if (primary_particle->status_[global_id] == DEAD) {
    #ifdef GGEMS_TRACKING
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] The particle id %d is dead!!!\n", global_id);
    }
    #endif
    return;
  }

  // Get the position and direction in local OBB coordinate
  GGfloat3 global_position = {primary_particle->px_[global_id], primary_particle->py_[global_id], primary_particle->pz_[global_id]};
  GGfloat3 global_direction = {primary_particle->dx_[global_id], primary_particle->dy_[global_id], primary_particle->dz_[global_id]};
  GGfloat3 local_position = GlobalToLocalPosition(&solid_box_data->obb_geometry_.matrix_transformation_, &global_position);
  GGfloat3 local_direction = GlobalToLocalDirection(&solid_box_data->obb_geometry_.matrix_transformation_, &global_direction);

  // Get borders of OBB
  GGfloat3 border_min = solid_box_data->obb_geometry_.border_min_xyz_;
  GGfloat3 border_max = solid_box_data->obb_geometry_.border_max_xyz_;

  // Get box size of solid box
  GGfloat3 box_size = {
    solid_box_data->box_size_xyz_[0],
    solid_box_data->box_size_xyz_[1],
    solid_box_data->box_size_xyz_[2]
  };

  // Get virtual element size
  GGint3 virtual_element_number = {
    solid_box_data->virtual_element_number_xyz_[0],
    solid_box_data->virtual_element_number_xyz_[1],
    solid_box_data->virtual_element_number_xyz_[2]
  };

  // Track particle until out of solid
  do {
    // Find next discrete photon interaction
    GetPhotonNextInteraction(primary_particle, random, particle_cross_sections, 0, global_id);
    GGfloat next_interaction_distance = primary_particle->next_interaction_distance_[global_id];
    GGchar next_discrete_process = primary_particle->next_discrete_process_[global_id];

    // Get safety position of particle to be sure particle is inside voxel
    TransportGetSafetyInsideAABB(
      &local_position,
      border_min.x, border_max.x,
      border_min.y, border_max.y,
      border_min.z, border_max.z,
      GEOMETRY_TOLERANCE
    );

    // Get the distance to next boundary
    GGfloat distance_to_next_boundary = ComputeDistanceToAABB(
      &local_position, &local_direction,
      border_min.x, border_max.x,
      border_min.y, border_max.y,
      border_min.z, border_max.z,
      GEOMETRY_TOLERANCE
    );

    // If distance to next boundary is inferior to distance to next interaction we move particle to boundary
    if (distance_to_next_boundary <= next_interaction_distance) {
      next_interaction_distance = distance_to_next_boundary + GEOMETRY_TOLERANCE;
      next_discrete_process = TRANSPORTATION;
    }

    #ifdef GGEMS_TRACKING
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Particle id: %d\n", global_id);
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Particle type: ");
      if (primary_particle->pname_[global_id] == PHOTON) printf("gamma\n");
      else if (primary_particle->pname_[global_id] == ELECTRON) printf("e-\n");
      else if (primary_particle->pname_[global_id] == POSITRON) printf("e+\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Local position (x, y, z): %e %e %e mm\n", local_position.x/mm, local_position.y/mm, local_position.z/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Local direction (x, y, z): %e %e %e\n", local_direction.x, local_direction.y, local_direction.z);
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Energy: %e keV\n", primary_particle->E_[global_id]/keV);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Solid id: %u\n", solid_box_data->solid_id_);
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Solid X Borders: %e %e mm\n", border_min.x/mm, border_max.x/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Solid Y Borders: %e %e mm\n", border_min.y/mm, border_max.y/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Solid Z Borders: %e %e mm\n", border_min.z/mm, border_max.z/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Material in voxel: %s\n", particle_cross_sections->material_names_[0]);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Next process: ");
      if (next_discrete_process == COMPTON_SCATTERING) printf("COMPTON_SCATTERING\n");
      if (next_discrete_process == PHOTOELECTRIC_EFFECT) printf("PHOTOELECTRIC_EFFECT\n");
      if (next_discrete_process == RAYLEIGH_SCATTERING) printf("RAYLEIGH_SCATTERING\n");
      if (next_discrete_process == TRANSPORTATION) printf("TRANSPORTATION\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_solid_box] Next interaction distance: %e mm\n", next_interaction_distance/mm);
    }
    #endif

    // Moving particle to next postion
    local_position = local_position + local_direction*next_interaction_distance;

    // Get safety position of particle to be sure particle is outside voxel
    TransportGetSafetyOutsideAABB(
      &local_position,
      border_min.x, border_max.x,
      border_min.y, border_max.y,
      border_min.z, border_max.z,
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

    // Check thresold
    if (primary_particle->E_[global_id] < threshold) primary_particle->status_[global_id] = DEAD;

    // Resolve process if different of TRANSPORTATION
    if (next_discrete_process != TRANSPORTATION) {
      PhotonDiscreteProcess(primary_particle, random, materials, particle_cross_sections, 0, global_id);

      local_direction.x = primary_particle->dx_[global_id];
      local_direction.y = primary_particle->dy_[global_id];
      local_direction.z = primary_particle->dz_[global_id];

      #ifdef HISTOGRAM
      if (next_discrete_process == PHOTOELECTRIC_EFFECT || next_discrete_process == COMPTON_SCATTERING) {
        GGfloat3 element_size = box_size / convert_float3(virtual_element_number);
        GGint3 voxel_id = convert_int3((local_position - border_min) / element_size);

        atomic_add(&histogram[voxel_id.x + voxel_id.y * virtual_element_number.x], 1);
      }
      #endif
    }
  } while (primary_particle->status_[global_id] == ALIVE);

  // Convert to global position
  global_position = LocalToGlobalPosition(&solid_box_data->obb_geometry_.matrix_transformation_, &local_position);
  primary_particle->px_[global_id] = global_position.x;
  primary_particle->py_[global_id] = global_position.y;
  primary_particle->pz_[global_id] = global_position.z;

  // Convert to global direction
  global_direction = LocalToGlobalDirection(&solid_box_data->obb_geometry_.matrix_transformation_, &local_direction);
  primary_particle->dx_[global_id] = global_direction.x;
  primary_particle->dy_[global_id] = global_direction.y;
  primary_particle->dz_[global_id] = global_direction.z;
}
