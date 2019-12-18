#include "GGEMS/opencl/use_double_precision.hh"
#include "GGEMS/processes/primary_particles.hh"
#include "GGEMS/randoms/random.hh"
#include "GGEMS/randoms/kiss_engine.hh"
#include "GGEMS/opencl/types.hh"
#include "GGEMS/maths/matrix_functions.hh"

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

  if (kGlobalIndex==2) {
    printf("Particle ID: %d\n", kGlobalIndex);
    printf("Focal: %4.7f %4.7f %4.7f\n", focal_spot_size.x, focal_spot_size.y, focal_spot_size.z);
    printf("phi: %4.7f, theta: %4.7f, particle: %u\n", phi, theta, particle_name);
    printf("rotation: %4.7f %4.7f %4.7f\n", rotation.x, rotation.y, rotation.z);
    printf("global position: %4.7f %4.7f %4.7f\n", global_position.x, global_position.y, global_position.z);
    printf("direction: %4.7f %4.7f %4.7f\n", direction.x, direction.y, direction.z);
    printf("matrix transformation:\n");
    printf("[%4.7f %4.7f %4.7f %4.7f\n", p_matrix_transformation->m00_, p_matrix_transformation->m01_, p_matrix_transformation->m02_, p_matrix_transformation->m03_);
    printf(" %4.7f %4.7f %4.7f %4.7f\n", p_matrix_transformation->m10_, p_matrix_transformation->m11_, p_matrix_transformation->m12_, p_matrix_transformation->m13_);
    printf(" %4.7f %4.7f %4.7f %4.7f\n", p_matrix_transformation->m20_, p_matrix_transformation->m21_, p_matrix_transformation->m22_, p_matrix_transformation->m23_);
    printf(" %4.7f %4.7f %4.7f %4.7f]\n", p_matrix_transformation->m30_, p_matrix_transformation->m31_, p_matrix_transformation->m32_, p_matrix_transformation->m33_);
  }
}
