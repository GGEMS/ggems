/*!
  \file TrackThroughVoxelizedSolid.cl

  \brief OpenCL kernel tracking particles within voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday June 16, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/materials/GGEMSMaterialsStack.hh"
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"
#include "GGEMS/randoms/GGEMSRandomStack.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/navigators/GGEMSPhotonNavigator.hh"

/*!
  \fn __kernel void track_through_voxelized_solid(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, __global GGEMSVoxelizedSolidData* voxelized_solid_data, __global GGEMSParticleCrossSections* particle_cross_sections, __global GGEMSMaterialTables* materials)
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param random - pointer on random numbers
  \param voxelized_solid_data - pointer to voxelized solid data
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param materials - pointer on material in navigator
  \brief OpenCL kernel tracking particles within voxelized solid
  \return no returned value
*/
__kernel void track_through_voxelized_solid(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  __global GGEMSVoxelizedSolidData const* voxelized_solid_data,
  __global GGuchar const* label_data,
  __global GGEMSParticleCrossSections const* particle_cross_sections,
  __global GGEMSMaterialTables const* materials)
{
  // Getting index of thread
  GGint const kParticleID = get_global_id(0);

  // Checking status of particle
  if (primary_particle->status_[kParticleID] == DEAD) return;

  // Checking if the current navigator is the selected navigator
  if (primary_particle->navigator_id_[kParticleID] != voxelized_solid_data->navigator_id_) return;

  // Position of particle
  GGfloat3 position;
  position.x = primary_particle->px_[kParticleID];
  position.y = primary_particle->py_[kParticleID];
  position.z = primary_particle->pz_[kParticleID];

  // Direction of particle
  GGfloat3 direction;
  direction.x = primary_particle->dx_[kParticleID];
  direction.y = primary_particle->dy_[kParticleID];
  direction.z = primary_particle->dz_[kParticleID];

  // Get tolerance of navigator
  GGfloat const kTolerance = voxelized_solid_data->tolerance_;

  // Get index of voxelized phantom, x, y, z and w (global index)
  GGint4 const kIndexVoxel;
  kIndexVoxel.x = (GGint)((position.x + voxelized_solid_data->position_xyz_.x) / voxelized_solid_data->voxel_sizes_xyz_.x);
  kIndexVoxel.y = (GGint)((position.y + voxelized_solid_data->position_xyz_.y) / voxelized_solid_data->voxel_sizes_xyz_.y);
  kIndexVoxel.z = (GGint)((position.z + voxelized_solid_data->position_xyz_.z) / voxelized_solid_data->voxel_sizes_xyz_.z);
  kIndexVoxel.w = kIndexVoxel.x
    + kIndexVoxel.y * voxelized_solid_data->number_of_voxels_xyz_.x
    + kIndexVoxel.z * voxelized_solid_data->number_of_voxels_xyz_.x * voxelized_solid_data->number_of_voxels_xyz_.y;

  // Get the material that compose this volume
  GGuchar const kIndexMaterial = label_data[kIndexVoxel.w];

  // Find next discrete photon interaction
  GetPhotonNextInteraction(primary_particle, random, particle_cross_sections, kIndexMaterial, kParticleID);
  GGfloat next_interaction_distance = primary_particle->next_interaction_distance_[kParticleID];
  GGuchar next_discrete_process = primary_particle->next_discrete_process_[kParticleID];

  // Get the borders of the current voxel
  GGfloat const kXMinVoxel = kIndexVoxel.x*voxelized_solid_data->voxel_sizes_xyz_.x - voxelized_solid_data->position_xyz_.x;
  GGfloat const kYMinVoxel = kIndexVoxel.y*voxelized_solid_data->voxel_sizes_xyz_.y - voxelized_solid_data->position_xyz_.y;
  GGfloat const kZMinVoxel = kIndexVoxel.z*voxelized_solid_data->voxel_sizes_xyz_.z - voxelized_solid_data->position_xyz_.z;
  GGfloat const kXMaxVoxel = kXMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.x;
  GGfloat const kYMaxVoxel = kYMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.y;
  GGfloat const kZMaxVoxel = kZMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.z;

  // Get safety position of particle to be sure particle is inside voxel
  TransportGetSafetyInsideAABB(&position, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, kTolerance);

  // Get the distance to next boundary
  GGfloat const distance_to_next_boundary = ComputeDistanceToAABB(&position, &direction, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, kTolerance);

  // If distance to next boundary is inferior to distance to next interaction
  // we move particle to boundary
  if (distance_to_next_boundary <= next_interaction_distance) {
    next_interaction_distance = distance_to_next_boundary + kTolerance;
    next_discrete_process = TRANSPORTATION;
  }

  // Moving particle to next postion
  position = GGfloat3Add(position, GGfloat3Scale(direction, next_interaction_distance));

  // Get safety position of particle to be sure particle is outside voxel
  TransportGetSafetyOutsideAABB(&position, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, kTolerance);

  // Update TOF, true for photon only
  primary_particle->tof_[kParticleID] += next_interaction_distance * C_LIGHT;

  // Storing new position
  primary_particle->px_[kParticleID] = position.x;
  primary_particle->py_[kParticleID] = position.y;
  primary_particle->pz_[kParticleID] = position.z;

  // Checking if particle outside voxelized solid navigator

  printf("******\n");
  printf("TRACK THROUGH\n");
  printf("-> Navigator infos <-\n");
  printf("  Navigator: %u\n", voxelized_solid_data->navigator_id_);
  printf("  Nb voxels: %u %u %u\n", voxelized_solid_data->number_of_voxels_xyz_.x, voxelized_solid_data->number_of_voxels_xyz_.y, voxelized_solid_data->number_of_voxels_xyz_.z);
  printf("  Voxel size: %e %e %e mm\n", voxelized_solid_data->voxel_sizes_xyz_.x/mm, voxelized_solid_data->voxel_sizes_xyz_.y/mm, voxelized_solid_data->voxel_sizes_xyz_.z/mm);
  printf("  Border X: %e %e mm\n", voxelized_solid_data->border_min_xyz_.x/mm, voxelized_solid_data->border_max_xyz_.x/mm);
  printf("  Border Y: %e %e mm\n", voxelized_solid_data->border_min_xyz_.y/mm, voxelized_solid_data->border_max_xyz_.y/mm);
  printf("  Border Z: %e %e mm\n", voxelized_solid_data->border_min_xyz_.z/mm, voxelized_solid_data->border_max_xyz_.z/mm);
  printf("-> Particle infos <-\n");
  printf("  After Position: %e %e %e mm\n", position.x/mm, position.y/mm, position.z/mm);
  printf("  Direction: %e %e %e\n", direction.x, direction.y, direction.z);
  printf("  Distance to next boundary: %e mm\n", distance_to_next_boundary/mm);
  printf("-> Voxel infos <-\n");
  printf("  Index voxel: %d %d %d %d\n", kIndexVoxel.x, kIndexVoxel.y, kIndexVoxel.z, kIndexVoxel.w);
  printf("  Label voxel: %d\n", kIndexMaterial);
  printf("  Material name voxel: %s\n", particle_cross_sections->material_names_[kIndexMaterial]);
  printf("  X voxel borders: %e %e mm\n", kXMinVoxel/mm, kXMaxVoxel/mm);
  printf("  Y voxel borders: %e %e mm\n", kYMinVoxel/mm, kYMaxVoxel/mm);
  printf("  Z voxel borders: %e %e mm\n", kZMinVoxel/mm, kZMaxVoxel/mm);
  printf("-> Process infos <-\n");
  if (next_discrete_process == 0) printf("  Next discrete process name: COMPTON_SCATTERING\n");
  if (next_discrete_process == 1) printf("  Next discrete process name: PHOTOELECTRIC_EFFECT\n");
  if (next_discrete_process == 2) printf("  Next discrete process name: RAYLEIGH_SCATTERING\n");
  printf("  Next interaction distance: %e mm\n", next_interaction_distance/mm);
}
