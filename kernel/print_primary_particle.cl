#include "primary_particles.hh"
#include "prng.hh"

__kernel void print_primary_particle(
  __global PrimaryParticles* p_primary_particle)
{
  // Get the index of thread
  size_t const kGlobalIndex = get_global_id(0);

  // Check index
  if (kGlobalIndex >= p_primary_particle->number_of_primaries_) return;

  for (int i = 0; i < 100000; ++i) {
    prng_uniform(p_primary_particle, kGlobalIndex);
  }

  //if (kGlobalIndex==0) {
  //  printf("%u\n", p_primary_particle->number_of_primaries_);
  //  printf("%4.7f\n", prng_uniform(p_primary_particle, kGlobalIndex));
  //}
}
