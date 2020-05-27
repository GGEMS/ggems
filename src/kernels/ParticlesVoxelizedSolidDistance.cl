/*!
  \file ParticlesVoxelizedSolidDistance.cl

  \brief OpenCL kernel computing distance between voxelized solid and particles

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesdayt May 19, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"

/*!
  \fn __kernel void particles_voxelized_solid_distance(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param voxelized_solid_data - pointer to voxelized solid data
  \brief OpenCL kernel computing distance between particle and solid in navigator
*/
__kernel void particles_voxelized_solid_distance(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSVoxelizedSolidData* voxelized_solid_data)
{
  // Getting index of thread
  GGint const kGlobalIndex = get_global_id(0);

  // Checking if the particle - navigator is 0. If yes the particle is already in another navigator
  if (primary_particle->particle_navigator_distance_[kGlobalIndex] == 0.0f) return;

  // Position of particle
  GGfloat3 position;
  position.x = primary_particle->px_[kGlobalIndex];
  position.y = primary_particle->py_[kGlobalIndex];
  position.z = primary_particle->pz_[kGlobalIndex];

  // Direction of particle
  GGfloat3 direction;
  direction.x = primary_particle->dx_[kGlobalIndex];
  direction.y = primary_particle->dy_[kGlobalIndex];
  direction.z = primary_particle->dz_[kGlobalIndex];

  // Check if particle inside voxelized navigator, if yes distance is 0.0 and
  // not need to compute particle - navigator distance
  if (IsParticleInVoxelizedNavigator(&position, voxelized_solid_data)) {
    primary_particle->particle_navigator_distance_[kGlobalIndex] = 0.0f;
    primary_particle->navigator_id_[kGlobalIndex] = voxelized_solid_data->navigator_id_;
    return;
  }

  // Compute distance between particles and voxelized navigator
  GGfloat const kDistance = ComputeDistanceToVoxelizedNavigator(&position, &direction, voxelized_solid_data);

  // Check distance value with previous value. Store the minimum value
  if (kDistance < primary_particle->particle_navigator_distance_[kGlobalIndex]) {
    primary_particle->particle_navigator_distance_[kGlobalIndex] = kDistance;
    primary_particle->navigator_id_[kGlobalIndex] = voxelized_solid_data->navigator_id_;
  }

  // Get infos from particles
  if(kGlobalIndex == 0) {
    printf("****\n");
    printf("Particle id: %d\n", kGlobalIndex);
    printf("Position: %4.7f %4.7f %4.7f\n", position.x, position.y, position.z);
    printf("Direction: %4.7f %4.7f %4.7f\n", direction.x, direction.y, direction.z);
    printf("Navigator id: %d\n", voxelized_solid_data->navigator_id_);
    printf("Particle - Navigator distance: %4.7f\n", primary_particle->particle_navigator_distance_[kGlobalIndex]);
    printf("Voxelized solid data:\n");
    printf("x border: %4.7f %4.7f mm\n", voxelized_solid_data->border_min_xyz_.x, voxelized_solid_data->border_max_xyz_.x);
    printf("y border: %4.7f %4.7f mm\n", voxelized_solid_data->border_min_xyz_.y, voxelized_solid_data->border_max_xyz_.y);
    printf("z border: %4.7f %4.7f mm\n", voxelized_solid_data->border_min_xyz_.z, voxelized_solid_data->border_max_xyz_.z);
  }
}
