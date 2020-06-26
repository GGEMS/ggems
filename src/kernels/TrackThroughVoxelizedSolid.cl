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

  // Get index of voxelized phantom, x, y, z and w (global index)
  GGint4 index_voxel;
  index_voxel.x = (GGint)((position.x + voxelized_solid_data->position_xyz_.x) / voxelized_solid_data->voxel_sizes_xyz_.x);
  index_voxel.y = (GGint)((position.y + voxelized_solid_data->position_xyz_.y) / voxelized_solid_data->voxel_sizes_xyz_.y);
  index_voxel.z = (GGint)((position.z + voxelized_solid_data->position_xyz_.z) / voxelized_solid_data->voxel_sizes_xyz_.z);
  index_voxel.w = index_voxel.x
    + index_voxel.y * voxelized_solid_data->number_of_voxels_xyz_.x
    + index_voxel.z * voxelized_solid_data->number_of_voxels_xyz_.x * voxelized_solid_data->number_of_voxels_xyz_.y;

  // Get the material that compose this volume
  GGuchar const kIndexMaterial = label_data[index_voxel.w];

  // Find next discrete photon interaction
  GetPhotonNextInteraction(primary_particle, random, particle_cross_sections, kIndexMaterial, kParticleID);
  GGfloat const kNextInteractionDistance = primary_particle->next_interaction_distance_[kParticleID];
  GGuchar const kNextDiscreteProcess = primary_particle->next_discrete_process_[kParticleID];

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
  printf("  Position: %e %e %e mm\n", position.x/mm, position.y/mm, position.z/mm);
  printf("  Direction: %e %e %e\n", direction.x, direction.y, direction.z);
  printf("-> Voxel infos <-\n");
  printf("Index voxel: %d %d %d %d\n", index_voxel.x, index_voxel.y, index_voxel.z, index_voxel.w);
  printf("Label voxel: %d\n", kIndexMaterial);
  printf("Material name voxel: %s\n", particle_cross_sections->material_names_[kIndexMaterial]);
  printf("-> Process infos <-\n");
  if (kNextDiscreteProcess == 0) printf("  Next discrete process name: COMPTON_SCATTERING\n");
  if (kNextDiscreteProcess == 1) printf("  Next discrete process name: PHOTOELECTRIC_EFFECT\n");
  if (kNextDiscreteProcess == 2) printf("  Next discrete process name: RAYLEIGH_SCATTERING\n");
  printf("  Next interaction distance: %e mm\n", kNextInteractionDistance/mm);
}
