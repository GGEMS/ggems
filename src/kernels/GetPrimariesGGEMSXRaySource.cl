#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/processes/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/randoms/GGEMSRandomStack.hh"
#include "GGEMS/randoms/GGEMSKissEngine.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"
#include "GGEMS/global/GGEMSConstants.hh"

__kernel void get_primaries_ggems_xray_source(
  __global GGEMSPrimaryParticles* p_primary_particle,
  __global GGEMSRandom* p_random,
  GGuchar const particle_name,
  __global GGdouble const* p_energy_spectrum,
  __global GGdouble const* p_cdf,
  GGuint const number_of_energy_bins,
  GGfloat const aperture,
  GGfloat3 const focal_spot_size,
  __global GGfloat44 const* p_matrix_transformation)
{
  // Get the index of thread
  GGint const kGlobalIndex = get_global_id(0);

  // Get random angles
  GGdouble phi = kiss_uniform(p_random, kGlobalIndex);
  GGdouble theta = kiss_uniform(p_random, kGlobalIndex);
  GGdouble const kAperture = 1.0 - cos((GGdouble)aperture);
  phi += PI_TWICE;
  theta = acos((GGdouble)1.0 - kAperture*theta);

  // Compute rotation
  GGfloat3 rotation = {
    cos(phi) * sin(theta),
    sin(phi) * sin(theta),
    cos(theta)
  };

  // Get direction of the cone beam. The beam is targeted to the isocenter, then
  // the direction is directly related to the position of the source.
  GGfloat3 global_position = LocalToGlobalPosition(p_matrix_transformation,
    MakeFloat3Zeros());
  GGfloat3 direction = GGfloat3UnitVector(
    GGfloat3Sub(MakeFloat3Zeros(), global_position));

  // Apply deflection (global coordinate)
  direction = RotateUz(rotation, direction);
  direction = GGfloat3UnitVector(direction);

  // Postition with focal (local)
  global_position.x = focal_spot_size.x
    * (kiss_uniform(p_random, kGlobalIndex) - 0.5f);
  global_position.y = focal_spot_size.y
    * (kiss_uniform(p_random, kGlobalIndex) - 0.5f);
  global_position.z = focal_spot_size.z
    * (kiss_uniform(p_random, kGlobalIndex) - 0.5f);

  // Apply transformation (local to global frame)
  global_position = LocalToGlobalPosition(p_matrix_transformation,
    global_position);

  // Getting a random energy
  GGfloat rndm_for_energy = kiss_uniform(p_random, kGlobalIndex);

  // Get index in cdf
  GGuint index_for_energy = BinarySearchLeft(
    rndm_for_energy,
    p_cdf,
    number_of_energy_bins,
    0,
    0
  );

  // Setting the energy for particles
  if (index_for_energy == number_of_energy_bins - 1) {
    p_primary_particle->p_E_[kGlobalIndex] =
      p_energy_spectrum[index_for_energy];
  }
  else
  {
    p_primary_particle->p_E_[kGlobalIndex] = LinearInterpolation(
      p_cdf[index_for_energy],
      p_energy_spectrum[index_for_energy],
      p_cdf[index_for_energy + 1],
      p_energy_spectrum[index_for_energy + 1],
      rndm_for_energy
    );
  }

  // Then set the mandatory field to create a new particle
  p_primary_particle->p_px_[kGlobalIndex] = global_position.x;
  p_primary_particle->p_py_[kGlobalIndex] = global_position.y;
  p_primary_particle->p_pz_[kGlobalIndex] = global_position.z;

  p_primary_particle->p_dx_[kGlobalIndex] = direction.x;
  p_primary_particle->p_dy_[kGlobalIndex] = direction.y;
  p_primary_particle->p_dz_[kGlobalIndex] = direction.z;

  p_primary_particle->p_tof_[kGlobalIndex] = 0.0f;
  p_primary_particle->p_status_[kGlobalIndex] = ALIVE;

  p_primary_particle->p_level_[kGlobalIndex] = PRIMARY;
  p_primary_particle->p_pname_[kGlobalIndex] = particle_name;

  p_primary_particle->p_geometry_id_[kGlobalIndex] = 0;
  p_primary_particle->p_next_discrete_process_[kGlobalIndex] = NO_PROCESS;
  p_primary_particle->p_next_interaction_distance_[kGlobalIndex] = 0.0f;
  p_primary_particle->p_scatter_order_[ kGlobalIndex ] = 0;
}
