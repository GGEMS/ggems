/*!
  \file ProjectToVoxelizedSolid.cl

  \brief OpenCL kernel moving particles to voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday May 29, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"

/*!
  \fn __kernel void project_to_voxelized_solid(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param voxelized_solid_data - pointer to voxelized solid data
  \brief OpenCL kernel moving particles to voxelized solid
  \return no returned value
*/
__kernel void project_to_voxelized_solid(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSVoxelizedSolidData const* voxelized_solid_data)
{
  // Getting index of thread
  GGint const kParticleID = get_global_id(0);

  // Checking if distance to navigator is OUT_OF_WORLD after computation distance
  // If yes, the particle is OUT_OF_WORLD and DEAD, so no tracking
  if (primary_particle->particle_navigator_distance_[kParticleID] == OUT_OF_WORLD) {
    primary_particle->status_[kParticleID] = DEAD;
    return;
  }

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

  // Distance to current navigator and geometry tolerance
  GGfloat const kDistance = primary_particle->particle_navigator_distance_[kParticleID];
  GGfloat const kTolerance = voxelized_solid_data->tolerance_;

  // Moving the particle slightly inside the volume
  position = GGfloat3Add(position, GGfloat3Scale(direction, kDistance + kTolerance));

  // Correcting the particle position if not totally inside due to float tolerance
  TransportGetSafetyInsideVoxelizedNavigator(&position, voxelized_solid_data);

  printf("******\n");
  printf("PROJECT TO\n");
  printf("Current navigator: %u\n", voxelized_solid_data->navigator_id_);
  printf("Selected Navigator: %u\n", primary_particle->navigator_id_[kParticleID]);
  printf("Position:\n");
  printf("    Before: %4.7f %4.7f %4.7f mm\n", primary_particle->px_[kParticleID], primary_particle->py_[kParticleID], primary_particle->pz_[kParticleID]);
  printf("    After: %4.7f %4.7f %4.7f mm\n", position.x, position.y, position.z);

  // Set new value for particles
  primary_particle->px_[kParticleID] = position.x;
  primary_particle->py_[kParticleID] = position.y;
  primary_particle->pz_[kParticleID] = position.z;

  //primary_particle->geometry_id_[kParticleID] = 0;
  primary_particle->tof_[kParticleID] += kDistance * C_LIGHT; // True only for photons !!!
}
