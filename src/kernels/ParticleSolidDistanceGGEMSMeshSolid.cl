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
  \file ParticleSolidDistanceGGEMSMeshSolid.cl

  \brief OpenCL kernel computing distance between mesh solid and particles

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday September 25, 2024
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"

#include "GGEMS/geometries/GGEMSMeshedSolidData.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"

/*!
  \fn kernel void particle_solid_distance_ggems_meshed_solid(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGEMSMeshedSolidData const* meshed_solid_data)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param meshed_solid_data - pointer to meshed solid data
  \brief OpenCL kernel computing distance between meshed solid and particles
*/
kernel void particle_solid_distance_ggems_meshed_solid(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSMeshedSolidData const* meshed_solid_data
)
{
  // Getting index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // Checking particle status. If DEAD, the particle is not track
  if (primary_particle->status_[global_id] == DEAD) return;

  // Checking if the particle - solid is 0. If yes the particle is already in another navigator
  if (primary_particle->particle_solid_distance_[global_id] == 0.0f) return;

  // Position of particle
  GGfloat3 position = {
    primary_particle->px_[global_id],
    primary_particle->py_[global_id],
    primary_particle->pz_[global_id]
  };

  // Direction of particle
  GGfloat3 direction = {
    primary_particle->dx_[global_id],
    primary_particle->dy_[global_id],
    primary_particle->dz_[global_id]
  };

  // Check if particle inside meshed navigator, if yes distance is 0.0 and not need to compute particle - solid distance
  if (IsParticleInOBB(&position, &meshed_solid_data->obb_geometry_)) {
    #ifdef GGEMS_TRACKING
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] --------------------------------------------------------------------------------\n");
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Find a closest solid\n");
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Particle id: %d\n", global_id);
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Particle in meshed solid, id: %d\n", meshed_solid_data->solid_id_);
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Particle solid distance: 0.0\n");
    }
    #endif
    primary_particle->particle_solid_distance_[global_id] = 0.0f;
    primary_particle->solid_id_[global_id] = meshed_solid_data->solid_id_;
    return;
  }

  // Compute distance between particles and meshed navigator
  GGfloat distance = ComputeDistanceToOBB(&position, &direction, &meshed_solid_data->obb_geometry_);

  // Check distance value with previous value. Store the minimum value
  if (distance < primary_particle->particle_solid_distance_[global_id]) {
    #ifdef GGEMS_TRACKING
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] --------------------------------------------------------------------------------\n");
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Find a closest solid\n");
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Particle id: %d\n", global_id);
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Particle in meshed solid, id: %d\n", meshed_solid_data->solid_id_);
      printf("[GGEMS OpenCL kernel particle_solid_distance_ggems_meshed_solid] Particle solid distance: %e mm\n", distance/mm);
    }
    #endif
    primary_particle->particle_solid_distance_[global_id] = distance;
    primary_particle->solid_id_[global_id] = meshed_solid_data->solid_id_;
  }
}
