#include "primary_particles.cl"
#include "prng.cl"

__kernel void print_primary_particle(
  __global PrimaryParticles* p_primary_particle)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  // Check index
  if (kGlobalIndex >= p_primary_particle->number_of_primaries_) return;

  for (int i = 0; i < 10; ++i) {
    float uniform = prng_uniform(p_primary_particle, kGlobalIndex);

    if (kGlobalIndex == 0) {
      printf("Uniform: %4.12f\n", uniform);
    }
  }
}
