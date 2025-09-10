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
  \file GetPrimariesGGEMSVoxelizedSource.cl

  \brief OpenCL kernel generating primaries for voxelized source

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday September 2, 2025
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/randoms/GGEMSKissEngine.hh"
#include "GGEMS/maths/GGEMSReferentialTransformation.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"
#include "GGEMS/physics/GGEMSParticleConstants.hh"
#include "GGEMS/physics/GGEMSProcessConstants.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"

/*!
  \fn kernel void get_primaries_ggems_voxelized_source(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, GGchar const particle_name, global GGfloat const* energy_spectrum, global GGfloat const* energy_cdf, GGint const number_of_energy_bins, GGchar const is_interp, global GGint const* activity_index, global GGfloat const* activity_cdf, GGint const number_of_activity_bins, global GGEMSVoxelizedSolidData const* voxelized_solid_data, global GGfloat44 const* matrix_transformation)
  \param particle_id_limit - particle id limit
  \param primary_particle - buffer of primary particles
  \param random - buffer for random number
  \param particle_name - name of particle
  \param energy_spectrum - energy spectrum
  \param energy_cdf - cumulative derivative function for energy
  \param number_of_energy_bins - number of energy bins
  \param is_interp - linear interpolation of energy or not
  \param activity_index - index of cdf activity
  \param activity_cdf - cdf of activity
  \param number_of_activity_bins - number of bins for cdf activity
  \param voxelized_solid_data - voxelized data for source
  \param matrix_transformation - matrix storing information about axis
  \brief Generate primaries for voxelized source
*/
kernel void get_primaries_ggems_voxelized_source(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  GGchar const particle_name,
  global GGfloat const* energy_spectrum,
  global GGfloat const* energy_cdf,
  GGint const number_of_energy_bins,
  GGchar const is_interp,
  global GGint const* activity_index,
  global GGfloat const* activity_cdf,
  GGint const number_of_activity_bins,
  global GGEMSVoxelizedSolidData const* voxelized_solid_data,
  global GGfloat44 const* matrix_transformation
)
{
  // Get the index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // Get random angles
  GGdouble phi = KissUniform(random, global_id);
  GGdouble theta = KissUniform(random, global_id);

  phi *= (GGdouble)TWO_PI;
  theta = acos(1.0 - 2.0 * theta);

  // Compute direction
  GGfloat3 direction = {
    cos(phi) * sin(theta),
    sin(phi) * sin(theta),
    cos(theta)
  };

  // Getting a random position in activity voxel
  GGdouble rdnm_for_act = KissUniform(random, global_id);

  // Getting index of cdf
  GGint index_for_act = BinarySearchLeft(rdnm_for_act, activity_cdf, number_of_activity_bins, 0, 0);
  // Getting index of vox. source
  GGint index_vox_src_total = activity_index[index_for_act];

  // Convert total index to a X, Y, Z index
  GGint3 index_vox;
  GGint slice_number_of_voxels = voxelized_solid_data->number_of_voxels_xyz_.x * voxelized_solid_data->number_of_voxels_xyz_.y;
  GGfloat3 voxel_size = voxelized_solid_data->voxel_sizes_xyz_;
  index_vox.z = index_vox_src_total / slice_number_of_voxels;
  index_vox.x = (index_vox_src_total - index_vox.z*slice_number_of_voxels) % (voxelized_solid_data->number_of_voxels_xyz_.x);
  index_vox.y = (index_vox_src_total - index_vox.z*slice_number_of_voxels) / (voxelized_solid_data->number_of_voxels_xyz_.x);

  // Get the border of whole voxelized source
  GGfloat3 border_min_vox_src = voxelized_solid_data->obb_geometry_.border_min_xyz_;
  GGfloat3 border_max_vox_src = voxelized_solid_data->obb_geometry_.border_max_xyz_;

  // Get the border of selected voxel
  GGfloat3 voxel_border_min = border_min_vox_src + convert_float3(index_vox) * voxel_size;
  GGfloat3 voxel_border_max = voxel_border_min + voxel_size;

  // Computing a random local_position in voxel and check vertex position is inside voxel
  GGfloat3 local_position;
  local_position.x = border_min_vox_src.x + (index_vox.x + KissUniform(random, global_id)) * voxel_size.x;
  local_position.y = border_min_vox_src.y + (index_vox.y + KissUniform(random, global_id)) * voxel_size.y;
  local_position.z = border_min_vox_src.z + (index_vox.z + KissUniform(random, global_id)) * voxel_size.z;

  // Check vertex position
  TransportGetSafetyInsideAABB(
    &local_position,
    voxel_border_min.x, voxel_border_max.x,
    voxel_border_min.y, voxel_border_max.y,
    voxel_border_min.z, voxel_border_max.z,
    GEOMETRY_TOLERANCE
  );

  GGfloat3 global_position = LocalToGlobalPosition(matrix_transformation, &local_position);

  GGint index_for_energy = 0;
  GGfloat rndm_for_energy = 0.0f;
  // For non monoenergy
  if (number_of_energy_bins != 1) {
    rndm_for_energy = KissUniform(random, global_id);
    index_for_energy = BinarySearchLeft(rndm_for_energy, energy_cdf, number_of_energy_bins+1, 0, 0);
  }

  // Setting the energy for particles
  primary_particle->E_[global_id] = (!is_interp) ?
    energy_spectrum[index_for_energy] :
    LinearInterpolation(
      energy_cdf[index_for_energy],
      energy_spectrum[index_for_energy],
      energy_cdf[index_for_energy + 1],
      energy_spectrum[index_for_energy + 1], rndm_for_energy
    );

  // Then set the mandatory field to create a new particle
  primary_particle->px_[global_id] = global_position.x;
  primary_particle->py_[global_id] = global_position.y;
  primary_particle->pz_[global_id] = global_position.z;

  primary_particle->dx_[global_id] = direction.x;
  primary_particle->dy_[global_id] = direction.y;
  primary_particle->dz_[global_id] = direction.z;

  primary_particle->scatter_[global_id] = FALSE;

  primary_particle->status_[global_id] = ALIVE;

  primary_particle->level_[global_id] = PRIMARY;
  primary_particle->pname_[global_id] = particle_name;

  primary_particle->particle_solid_distance_[global_id] = OUT_OF_WORLD;
  primary_particle->next_discrete_process_[global_id] = NO_PROCESS;
  primary_particle->next_interaction_distance_[global_id] = 0.0f;

  #ifdef OPENGL
  // Storing vertex position for OpenGL
  if (global_id < MAXIMUM_DISPLAYED_PARTICLES) {
    primary_particle->stored_particles_gl_[global_id] = 0;

    for (GGint i = 0; i < MAXIMUM_INTERACTIONS; ++i) {
      primary_particle->px_gl_[global_id*MAXIMUM_INTERACTIONS+i] = 0.0f;
      primary_particle->py_gl_[global_id*MAXIMUM_INTERACTIONS+i] = 0.0f;
      primary_particle->pz_gl_[global_id*MAXIMUM_INTERACTIONS+i] = 0.0f;
    }

    primary_particle->px_gl_[global_id*MAXIMUM_INTERACTIONS] = primary_particle->px_[global_id];
    primary_particle->py_gl_[global_id*MAXIMUM_INTERACTIONS] = primary_particle->py_[global_id];
    primary_particle->pz_gl_[global_id*MAXIMUM_INTERACTIONS] = primary_particle->pz_[global_id];

    // Storing final index
    primary_particle->stored_particles_gl_[global_id] = 1;
  }
  #endif

  #ifdef GGEMS_TRACKING
  if (global_id == primary_particle->particle_tracking_id) {
    printf("[GGEMS OpenCL kernel get_primaries_ggems_voxelized_source] ################################################################################\n");
    printf("[GGEMS OpenCL kernel get_primaries_ggems_voxelized_source] Particle id: %d\n", global_id);
    printf("[GGEMS OpenCL kernel get_primaries_ggems_voxelized_source] Particle type: ");
    if (primary_particle->pname_[global_id] == PHOTON) printf("gamma\n");
    else if (primary_particle->pname_[global_id] == ELECTRON) printf("e-\n");
    else if (primary_particle->pname_[global_id] == POSITRON) printf("e+\n");
    printf("[GGEMS OpenCL kernel get_primaries_ggems_voxelized_source] Position (x, y, z): %e %e %e mm\n", global_position.x/mm, global_position.y/mm, global_position.z/mm);
    printf("[GGEMS OpenCL kernel get_primaries_ggems_voxelized_source] Direction (x, y, z): %e %e %e\n", direction.x, direction.y, direction.z);
    printf("[GGEMS OpenCL kernel get_primaries_ggems_voxelized_source] Energy: %e keV\n", primary_particle->E_[global_id]/keV);
  }
  #endif
}
