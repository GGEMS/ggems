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
#include "GGEMS/physics/GGEMSParticleConstants.hh"
#include "GGEMS/physics/GGEMSProcessConstants.hh"

/*!
  \fn __kernel void get_primaries_ggems_xray_source(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, GGuchar const particle_name, __global GGfloat const* energy_spectrum, __global GGfloat const* cdf, GGuint const number_of_energy_bins, GGfloat const aperture, GGfloat3 const focal_spot_size, __global GGfloat44 const* matrix_transformation)
  \param primary_particle - buffer of primary particles
  \param random - buffer for random number
  \param particle_name - name of particle
  \param energy_spectrum - energy spectrum
  \param cdf - cumulative derivative function
  \param number_of_energy_bins - number of energy bins
  \param aperture - source aperture
  \param focal_spot_size - focal spot size of xray-source
  \param matrix_transformation - matrix storing information about axis
  \brief Generate primaries for xray source
  \return no returned value
*/
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
  GGint const kParticleID = get_global_id(0);

  // Get random angles
  GGdouble phi = KissUniform(random, kParticleID);
  GGdouble theta = KissUniform(random, kParticleID);
  GGdouble const kAperture = 1.0 - cos((GGdouble)aperture);
  phi += TWO_PI;
  theta = acos(1.0 - kAperture*theta);

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
  global_position.x = focal_spot_size.x * (KissUniform(random, kParticleID) - 0.5f);
  global_position.y = focal_spot_size.y * (KissUniform(random, kParticleID) - 0.5f);
  global_position.z = focal_spot_size.z * (KissUniform(random, kParticleID) - 0.5f);

  // Apply transformation (local to global frame)
  global_position = LocalToGlobalPosition(matrix_transformation, global_position);

  // Getting a random energy
  GGfloat rndm_for_energy = KissUniform(random, kParticleID);

  // Get index in cdf
  GGuint index_for_energy = BinarySearchLeft(rndm_for_energy, cdf, number_of_energy_bins, 0, 0);

  // Setting the energy for particles
  if (index_for_energy == number_of_energy_bins - 1) {
    primary_particle->E_[kParticleID] = energy_spectrum[index_for_energy];
  }
  else {
    primary_particle->E_[kParticleID] = LinearInterpolation(cdf[index_for_energy], energy_spectrum[index_for_energy], cdf[index_for_energy + 1], energy_spectrum[index_for_energy + 1], rndm_for_energy);
  }

  // Then set the mandatory field to create a new particle
  primary_particle->px_[kParticleID] = global_position.x;
  primary_particle->py_[kParticleID] = global_position.y;
  primary_particle->pz_[kParticleID] = global_position.z;

  primary_particle->dx_[kParticleID] = direction.x;
  primary_particle->dy_[kParticleID] = direction.y;
  primary_particle->dz_[kParticleID] = direction.z;

  primary_particle->tof_[kParticleID] = 0.0f;
  primary_particle->status_[kParticleID] = ALIVE;

  primary_particle->level_[kParticleID] = PRIMARY;
  primary_particle->pname_[kParticleID] = particle_name;

  //primary_particle->geometry_id_[kParticleID] = 0;
  primary_particle->particle_navigator_distance_[kParticleID] = OUT_OF_WORLD;
  primary_particle->next_discrete_process_[kParticleID] = NO_PROCESS;
  primary_particle->next_interaction_distance_[kParticleID] = 0.0f;
  //primary_particle->scatter_order_[kParticleID] = 0;
}
