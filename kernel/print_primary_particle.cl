typedef struct __attribute__((aligned (1))) PrimaryParticles_t
{
  unsigned int p_prng_state_1_[861635]; /*!< State 1 of the prng in unsigned int 32 */
  unsigned int p_prng_state_2_[861635]; /*!< State 2 of the prng in unsigned int 32 */
  unsigned int p_prng_state_3_[861635]; /*!< State 3 of the prng in unsigned int 32 */
  unsigned int p_prng_state_4_[861635]; /*!< State 4 of the prng in unsigned int 32 */
  unsigned int p_prng_state_5_[861635];
  size_t number_of_primaries_;
}PrimaryParticles;

#include "prng.hh"

__kernel void print_primary_particle(
  __global PrimaryParticles* p_primary_particle)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  // Check index
  if (kGlobalIndex >= 861635) return;

  if (kGlobalIndex == 10) {
    printf("%d %d %d %d %d %d %4.7f\n", kGlobalIndex,
      p_primary_particle->p_prng_state_1_[kGlobalIndex],
      p_primary_particle->p_prng_state_2_[kGlobalIndex],
      p_primary_particle->p_prng_state_3_[kGlobalIndex],
      p_primary_particle->p_prng_state_4_[kGlobalIndex],
      p_primary_particle->p_prng_state_5_[kGlobalIndex],
      prng_uniform(p_primary_particle));
  }

  // Store the state of random
  /*RandomState rndm_state;
  rndm_state.p_prng_state_1_ = &p_prng_state_1[kGlobalIndex];
  rndm_state.p_prng_state_2_ = &p_prng_state_2[kGlobalIndex];
  rndm_state.p_prng_state_3_ = &p_prng_state_3[kGlobalIndex];
  rndm_state.p_prng_state_4_ = &p_prng_state_4[kGlobalIndex];
  rndm_state.p_prng_state_5_ = &p_prng_state_5[kGlobalIndex];

  if (kGlobalIndex==15) {
    printf("---- BEFORE:\n");
    printf("%d\n", p_prng_state_1[kGlobalIndex]);
    printf("%d\n", p_prng_state_2[kGlobalIndex]);
    printf("%d\n", p_prng_state_3[kGlobalIndex]);
    printf("%d\n", p_prng_state_4[kGlobalIndex]);
    printf("%d\n", p_prng_state_5[kGlobalIndex]);

    printf("%4.7f\n", prng_uniform(&rndm_state));

    printf("---- AFTER:\n");
    printf("%d\n", p_prng_state_1[kGlobalIndex]);
    printf("%d\n", p_prng_state_2[kGlobalIndex]);
    printf("%d\n", p_prng_state_3[kGlobalIndex]);
    printf("%d\n", p_prng_state_4[kGlobalIndex]);
    printf("%d\n", p_prng_state_5[kGlobalIndex]);
  }*/



 // if (kGlobalIndex == index) {
 //   printf("Index: %d, float: %d, toto: %d\n", kGlobalIndex,
 //     p_prng_state_1[kGlobalIndex], toto(p_prng_state_1[kGlobalIndex]));
 // }
}
