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
  \file ParticleSolidDistanceGGEMSVoxelizedSolid.cl

  \brief OpenCL kernel computing distance between voxelized solid and particles

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday May 19, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"

/*!
  \fn __kernel void particle_solid_distance_ggems_voxelized_solid(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param voxelized_solid_data - pointer to voxelized solid data
  \brief OpenCL kernel computing distance between voxelized solid and particles
  \return no returned value
*/
__kernel void particle_solid_distance_ggems_voxelized_solid(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSVoxelizedSolidData const* voxelized_solid_data)
{
  // Getting index of thread
  GGint const kParticleID = get_global_id(0);

  // Checking particle status. If DEAD, the particle is not track
  if (primary_particle->status_[kParticleID] == DEAD) return;

  // Checking if the particle - solid is 0. If yes the particle is already in another navigator
  if (primary_particle->particle_solid_distance_[kParticleID] == 0.0f) return;

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

  // Check if particle inside voxelized navigator, if yes distance is 0.0 and not need to compute particle - solid distance
  if (IsParticleInVoxelizedSolid(&position, voxelized_solid_data)) {
    primary_particle->particle_solid_distance_[kParticleID] = 0.0f;
    primary_particle->solid_id_[kParticleID] = voxelized_solid_data->solid_id_;
    return;
  }

  // // Compute distance between particles and voxelized navigator
  // GGfloat const kDistance = ComputeDistanceToVoxelizedNavigator(&position, &direction, voxelized_solid_data);

  // // Check distance value with previous value. Store the minimum value
  // if (kDistance < primary_particle->particle_solid_distance_[kParticleID]) {
  //   primary_particle->particle_solid_distance_[kParticleID] = kDistance;
  //   primary_particle->solid_id_[kParticleID] = voxelized_solid_data->solid_id_;
  // }
}
