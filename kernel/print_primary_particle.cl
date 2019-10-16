#include "primary_particles.cl"
#include "prng.cl"

__kernel void print_primary_particle(
  __global PrimaryParticles* p_primary_particle)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  // Check index
  if (kGlobalIndex >= p_primary_particle->number_of_primaries_) return;

  //for (int i = 0; i < 100000; ++i) {
  //  float uniform = prng_uniform(p_primary_particle, kGlobalIndex);
  //}

  //for (int i = 0; i < 100000; ++i) {
  //  float gaussian = prng_gaussian(p_primary_particle, kGlobalIndex, 2.5f);
 // }

  for (int i = 0; i < 100000; ++i) {
    unsigned int poisson = prng_poisson(p_primary_particle, kGlobalIndex, 5);
  }
}
