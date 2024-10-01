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
  \file ProjectToGGEMSMeshSolid.cl

  \brief OpenCL kernel moving particles to meshed solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday September 25, 2024
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/geometries/GGEMSMeshedSolidData.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"
#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"

/*!
  \fn kernel void project_to_ggems_meshed_solid(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGEMSMeshedSolidData const* meshed_solid_data)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param meshed_solid_data - pointer to meshed solid data
  \brief OpenCL kernel moving particles to meshed solid
*/
kernel void project_to_ggems_meshed_solid(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSMeshedSolidData const* meshed_solid_data
)
{
  // Getting index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // No solid detected, consider particle as dead
  if(primary_particle->solid_id_[global_id] == -1) primary_particle->status_[global_id] = DEAD;

  // Checking if distance to navigator is OUT_OF_WORLD after computation distance
  // If yes, the particle is OUT_OF_WORLD and DEAD, so no tracking
  if (primary_particle->particle_solid_distance_[global_id] == OUT_OF_WORLD) {
    primary_particle->solid_id_[global_id] = -1; // -1 is out_of_world, using for debugging
    primary_particle->status_[global_id] = DEAD;

    #ifdef OPENGL
    if (global_id < MAXIMUM_DISPLAYED_PARTICLES) {
      // Storing OpenGL index on OpenCL private memory
      GGint stored_particles_gl = primary_particle->stored_particles_gl_[global_id];

      // Checking if buffer is full
      if (stored_particles_gl != MAXIMUM_INTERACTIONS) {
        primary_particle->px_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] = primary_particle->px_[global_id] + primary_particle->dx_[global_id]*100.0*m;
        primary_particle->py_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] = primary_particle->py_[global_id] + primary_particle->dy_[global_id]*100.0*m;
        primary_particle->pz_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] = primary_particle->pz_[global_id] + primary_particle->dz_[global_id]*100.0*m;

        // Storing final index
        primary_particle->stored_particles_gl_[global_id] += 1;
      }
    }
    #endif

    return;
  }

  // Checking if the current navigator is the selected navigator
  if (primary_particle->solid_id_[global_id] != meshed_solid_data->solid_id_) return;

  // Checking status of particle
  if (primary_particle->status_[global_id] == DEAD) return;

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

  // Distance to current navigator and geometry tolerance
  GGfloat distance = primary_particle->particle_solid_distance_[global_id];

  // Moving the particle slightly inside the volume
  position += direction*(distance+GEOMETRY_TOLERANCE);

  // Correcting the particle position if not totally inside due to float tolerance
  TransportGetSafetyInsideOBB(&position, &meshed_solid_data->obb_geometry_);

  // Set new value for particles
  primary_particle->px_[global_id] = position.x;
  primary_particle->py_[global_id] = position.y;
  primary_particle->pz_[global_id] = position.z;

  primary_particle->particle_solid_distance_[global_id] = 0.0f;

  //printf("%d", global_id);
  #ifdef GGEMS_TRACKING
  if (global_id == primary_particle->particle_tracking_id) {
    printf("[GGEMS OpenCL kernel project_to_ggems_meshed_solid] ********************************************************************************\n");
    printf("[GGEMS OpenCL kernel project_to_ggems_meshed_solid] Project to closest solid\n");
    printf("[GGEMS OpenCL kernel project_to_ggems_meshed_solid] Particle id: %d\n", global_id);
    printf("[GGEMS OpenCL kernel project_to_ggems_meshed_solid] Position (x, y, z): %e %e %e mm\n", position.x/mm, position.y/mm, position.z/mm);
  }
  #endif
}
