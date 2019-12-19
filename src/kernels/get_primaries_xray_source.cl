#include "GGEMS/opencl/use_double_precision.hh"
#include "GGEMS/processes/primary_particles.hh"
#include "GGEMS/randoms/random.hh"
#include "GGEMS/randoms/kiss_engine.hh"
#include "GGEMS/maths/matrix_functions.hh"
#include "GGEMS/maths/math_functions.hh"
#include "GGEMS/global/ggems_constants.hh"

__kernel void get_primaries_xray_source(
  __global PrimaryParticles* p_primary_particle,
  __global Random* p_random,
  ucharcl_t const particle_name,
  __global double const* p_energy_spectrum,
  __global double const* p_cdf,
  uintcl_t const number_of_energy_bins,
  f32cl_t const aperture,
  f323cl_t const focal_spot_size,
  __global float4x4 const* p_matrix_transformation)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  // Get random angles
  f64cl_t phi = kiss_uniform(p_random, kGlobalIndex);
  f64cl_t theta = kiss_uniform(p_random, kGlobalIndex);
  f64cl_t const kAperture = 1.0 - cos((f64cl_t)aperture);
  phi += PI_TWICE;
  theta = acos((f64cl_t)1.0 - kAperture*theta);

  // Compute rotation
  f323cl_t rotation = {
    cos(phi) * sin(theta),
    sin(phi) * sin(theta),
    cos(theta)
  };

  // Get direction of the cone beam. The beam is targeted to the isocenter, then
  // the direction is directly related to the position of the source.
  f323cl_t global_position = LocalToGlobalPosition(p_matrix_transformation,
    MakeFloat3x1Zeros());
  f323cl_t direction = f323x1_unit(
    f323x1_sub(MakeFloat3x1Zeros(), global_position));

  // Apply deflection (global coordinate)
  direction = RotateUz(rotation, direction);
  direction = f323x1_unit(direction);

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
  f32cl_t rndm_for_energy = kiss_uniform(p_random, kGlobalIndex);

  // Get index in cdf
  uintcl_t index_for_energy = binary_search_left(
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
    p_primary_particle->p_E_[kGlobalIndex] = linear_interpolation(
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
