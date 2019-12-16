#include "GGEMS/opencl/use_double_precision.hh"
#include "GGEMS/processes/primary_particles.hh"
#include "GGEMS/randoms/random.hh"
#include "GGEMS/randoms/kiss_engine.hh"

__kernel void get_primaries_xray_source(
  __global PrimaryParticles* p_primary_particle,
  __global Random* p_random,
  __global double const* p_energy_spectrum,
  __global double const* p_cdf,
  unsigned int const number_of_energy_bins)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  if (kGlobalIndex == 2) {
    float const kUniform = kiss_uniform(p_random, kGlobalIndex);
    unsigned int const kPoisson = kiss_poisson(p_random, kGlobalIndex, 5);
    float const kGauss = kiss_gauss(p_random, kGlobalIndex, 2.4f);

    printf("Particle id: %d, uniform: %4.7f, poisson: %u, gauss: %4.7f\n",
      kGlobalIndex, kUniform, kPoisson, kGauss);
  }
}
