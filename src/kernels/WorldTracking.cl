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
  \file WorldTracking.cl

  \brief OpenCL kernel tracking particles through world

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday March 3, 2021
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/physics/GGEMSParticleConstants.hh"

/*!
  \fn kernel void world_tracking(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGint* photon_tracking, global GGDosiType* edep_tracking, global GGDosiType* momentum_x, global GGDosiType* momentum_y, global GGDosiType* momentum_z, GGsize width, GGsize height, GGsize depth, GGfloat size_x, GGfloat size_y, GGfloat size_z)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param photon_tracking - photon tracking counter in world
  \param edep_tracking - energy tracking in world
  \param momentum_x - sum of momentum along X
  \param momentum_y - sum of momentum along Y
  \param momentum_z - sum of momentum along Z
  \param width - number of elements in world along X
  \param height - number of elements in world along Y
  \param depth - number of elements in world along Z
  \param size_x - size of world voxel along X
  \param size_y - size of world voxel along Y
  \param size_z - size of world voxel along Z
  \brief tracking particles through world volume
*/
kernel void world_tracking(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGint* photon_tracking,
  global GGDosiType* edep_tracking,
  global GGDosiType* momentum_x,
  global GGDosiType* momentum_y,
  global GGDosiType* momentum_z,
  GGsize width,
  GGsize height,
  GGsize depth,
  GGfloat size_x,
  GGfloat size_y,
  GGfloat size_z
)
{
  // Getting index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  if (primary_particle->status_[global_id] == DEAD) return;

  // In world, the particles is tracked using a DDA algorithm
  // Get direction of particle
  GGfloat3 direction = {primary_particle->dx_[global_id], primary_particle->dy_[global_id], primary_particle->dz_[global_id]};
  // Get point x1, y1 and z1
  GGfloat3 p1 = {primary_particle->px_[global_id], primary_particle->py_[global_id], primary_particle->pz_[global_id]};
  // Get voxel size
  GGfloat3 size = {size_x, size_y, size_z};

  // Computing point x2, y2, z2
  GGfloat distance = primary_particle->particle_solid_distance_[global_id] == OUT_OF_WORLD ? 10000.0f : primary_particle->particle_solid_distance_[global_id];
  GGfloat3 p2 = p1 + distance*direction;

  // Start index
  GGint3 dim = {width, height, depth};
  GGint3 index = convert_int3((p1 - (size*convert_float3(dim)*-0.5f))/size);

  // Computing difference between p1 and p2 and length for each axis
  GGfloat3 diff_p1_p2 = p2 - p1;
  GGfloat3 len_p1_p2 = fabs(diff_p1_p2);

  // Getting main direction and size
  GGfloat length = len_p1_p2.x;
  GGfloat main_size = size.x;
  if (len_p1_p2.y > length) {
    length = len_p1_p2.y;
    main_size = size.y;
  }
  else if (len_p1_p2.z > length) {
    length = len_p1_p2.z;
    main_size = size.z;
  }

  GGfloat inv_length = 1.0f / length;

  // Computing incrementation
  GGfloat3 increment = inv_length*diff_p1_p2;

  // Computing number of step
  GGint step = (GGint)(1.0f + (length/main_size));
  GGint global_index_world = 0;

  for (GGint i = 0; i < step; ++i) {
    // Checking index
    if (index.x < 0 || index.x >= dim.x || index.y < 0 || index.y >= dim.y || index.z < 0 || index.z >= dim.z) {
      break;
    }

    global_index_world = index.x + index.y * dim.x + index.z * dim.x * dim.y;
    if (photon_tracking) atomic_add(&photon_tracking[global_index_world], 1);

    #ifdef DOSIMETRY_DOUBLE_PRECISION
    if (edep_tracking) AtomicAddDouble(&edep_tracking[global_index_world], (GGDosiType)primary_particle->E_[global_id]);
    if (momentum_x) AtomicAddDouble(&momentum_x[global_index_world], (GGDosiType)primary_particle->dx_[global_id]);
    if (momentum_y) AtomicAddDouble(&momentum_y[global_index_world], (GGDosiType)primary_particle->dy_[global_id]);
    if (momentum_z) AtomicAddDouble(&momentum_z[global_index_world], (GGDosiType)primary_particle->dz_[global_id]);
    #else
    if (edep_tracking) AtomicAddFloat(&edep_tracking[global_index_world], (GGDosiType)primary_particle->E_[global_id]);
    if (momentum_x) AtomicAddFloat(&momentum_x[global_index_world], (GGDosiType)primary_particle->dx_[global_id]);
    if (momentum_y) AtomicAddFloat(&momentum_y[global_index_world], (GGDosiType)primary_particle->dy_[global_id]);
    if (momentum_z) AtomicAddFloat(&momentum_z[global_index_world], (GGDosiType)primary_particle->dz_[global_id]);
    #endif

    p1 += increment*main_size;
    index = convert_int3((p1 - (size*convert_float3(dim)*-0.5f))/size);
  }

  #ifdef GGEMS_TRACKING
  if (global_id == primary_particle->particle_tracking_id) {
    printf("[GGEMS OpenCL kernel world_tracking] ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("[GGEMS OpenCL kernel world_tracking] World tracking particle\n");
    printf("[GGEMS OpenCL kernel world_tracking] Particle status: %d, DEAD: %d, ALIVE: %d\n", primary_particle->status_[global_id], DEAD, ALIVE);
    printf("[GGEMS OpenCL kernel world_tracking] Particle position: %e %e %e mm\n", primary_particle->px_[global_id]/mm, primary_particle->py_[global_id]/mm, primary_particle->pz_[global_id]/mm);
    printf("[GGEMS OpenCL kernel world_tracking] Particle direction: %e %e %e\n", primary_particle->dx_[global_id], primary_particle->dy_[global_id], primary_particle->dz_[global_id]);
    printf("[GGEMS OpenCL kernel world_tracking] Particle energy: %e keV\n", primary_particle->E_[global_id]/keV);
    printf("[GGEMS OpenCL kernel world_tracking] Distance to next solid: %e mm\n", distance/mm);
  }
  #endif
}
