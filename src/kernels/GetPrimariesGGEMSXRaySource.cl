/*!
  \file GetPrimariesGGEMSXRaySource.cl

  \brief OpenCL kernel generating primaries for X-Ray source

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/randoms/GGEMSKissEngine.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"

__kernel void get_primaries_ggems_xray_source(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  GGuchar const particle_name,
  __global GGfloat const* energy_spectrum,
  __global GGfloat const* cdf,
  GGuint const number_of_energy_bins,
  GGfloat const aperture,
  GGfloat3 const focal_spot_size,
  __global GGfloat44 const* matrix_transformation)
{
  // Get the index of thread
  GGint const kGlobalIndex = get_global_id(0);

  // Get random angles
  GGdouble phi = KissUniform(random, kGlobalIndex);
  GGdouble theta = KissUniform(random, kGlobalIndex);
  GGdouble const kAperture = 1.0 - cos((GGdouble)aperture);
  phi += TWO_PI;
  theta = acos((GGdouble)1.0 - kAperture*theta);

  // Compute rotation
  GGfloat3 rotation = {cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)};

  // Get direction of the cone beam. The beam is targeted to the isocenter, then
  // the direction is directly related to the position of the source.
  GGfloat3 global_position = LocalToGlobalPosition(matrix_transformation, MakeFloat3Zeros());
  GGfloat3 direction = GGfloat3UnitVector(GGfloat3Sub(MakeFloat3Zeros(), global_position));

  // Apply deflection (global coordinate)
  direction = RotateUnitZ(rotation, direction);
  direction = GGfloat3UnitVector(direction);

  // Postition with focal (local)
  global_position.x = focal_spot_size.x * (KissUniform(random, kGlobalIndex) - 0.5f);
  global_position.y = focal_spot_size.y * (KissUniform(random, kGlobalIndex) - 0.5f);
  global_position.z = focal_spot_size.z * (KissUniform(random, kGlobalIndex) - 0.5f);

  // Apply transformation (local to global frame)
  global_position = LocalToGlobalPosition(matrix_transformation, global_position);

  // Getting a random energy
  GGfloat rndm_for_energy = KissUniform(random, kGlobalIndex);

  // Get index in cdf
  GGuint index_for_energy = BinarySearchLeft(rndm_for_energy, cdf, number_of_energy_bins, 0, 0);

  // Setting the energy for particles
  if (index_for_energy == number_of_energy_bins - 1) {
    primary_particle->E_[kGlobalIndex] = energy_spectrum[index_for_energy];
  }
  else {
    primary_particle->E_[kGlobalIndex] = LinearInterpolation(cdf[index_for_energy], energy_spectrum[index_for_energy], cdf[index_for_energy + 1], energy_spectrum[index_for_energy + 1], rndm_for_energy);
  }

  // Then set the mandatory field to create a new particle
  primary_particle->px_[kGlobalIndex] = global_position.x;
  primary_particle->py_[kGlobalIndex] = global_position.y;
  primary_particle->pz_[kGlobalIndex] = global_position.z;

  primary_particle->dx_[kGlobalIndex] = direction.x;
  primary_particle->dy_[kGlobalIndex] = direction.y;
  primary_particle->dz_[kGlobalIndex] = direction.z;

  primary_particle->tof_[kGlobalIndex] = 0.0f;
  primary_particle->status_[kGlobalIndex] = ALIVE;

  primary_particle->level_[kGlobalIndex] = PRIMARY;
  primary_particle->pname_[kGlobalIndex] = particle_name;

  primary_particle->geometry_id_[kGlobalIndex] = 0;
  primary_particle->next_discrete_process_[kGlobalIndex] = NO_PROCESS;
  primary_particle->next_interaction_distance_[kGlobalIndex] = 0.0f;
  primary_particle->scatter_order_[ kGlobalIndex ] = 0;
}
