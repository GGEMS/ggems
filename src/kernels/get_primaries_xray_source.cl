#include "GGEMS/opencl/use_double_precision.hh"
#include "GGEMS/processes/primary_particles.hh"
#include "GGEMS/randoms/random.hh"
#include "GGEMS/randoms/kiss_engine.hh"
#include "GGEMS/opencl/types.hh"
#include "GGEMS/maths/matrix_types.hh"

__kernel void get_primaries_xray_source(
  __global PrimaryParticles* p_primary_particle,
  __global Random* p_random,
  ucharcl_t const particle_name,
  __global double const* p_energy_spectrum,
  __global double const* p_cdf,
  uintcl_t const number_of_energy_bins,
  f32cl_t const aperture)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  // Get random angle
  f64cl_t phi = kiss_uniform(p_random, kGlobalIndex);
  f64cl_t theta = kiss_uniform(p_random, kGlobalIndex);
  f64cl_t const kAperture = 1.0 - cos((f64cl_t)aperture);
  phi += PI_TWICE;
  theta = acos((f64cl_t)1.0 - kAperture*theta);

  if (kGlobalIndex==2) {
    printf("phi: %4.7f, theta: %4.7f, particle: %u\n", phi, theta, particle_name);
  }

  /*
  if (kGlobalIndex == 2) {
    float const kUniform = kiss_uniform(p_random, kGlobalIndex);
    unsigned int const kPoisson = kiss_poisson(p_random, kGlobalIndex, 5);
    float const kGauss = kiss_gauss(p_random, kGlobalIndex, 2.4f);

    printf("Particle id: %d, uniform: %4.7f, poisson: %u, gauss: %4.7f\n",
      kGlobalIndex, kUniform, kPoisson, kGauss);
  }*/
}
