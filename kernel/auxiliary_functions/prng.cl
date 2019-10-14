#ifndef GUARD_GGEMS_KERNEL_PRNG_CL
#define GUARD_GGEMS_KERNEL_PRNG_CL

/*!
  \file prng.cl

  \brief Function for pseudo random number generator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday October 10, 2019
*/

#include "system_of_units.cl"

/*!
  \fn inline float prng_uniform(__global PrimaryParticles* p_primary_particle, size_t const index)
  \param p_primary_particle - pointer on primary particles on device
  \param index - index of thread
  \brief JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder
  and the BigCrunch tests in TestU01
*/
inline float prng_uniform(__global PrimaryParticles* p_primary_particle,
  int const index)
{
  // y ^= (y<<5);
  // y ^= (y>>7);
  // y ^= (y<<22);
  // t = z+w+c;
  // z = w;
  // c = t < 0;
  // w = t & 2147483647;
  // x += 1411392427;

  // Get the index of primary particle
  p_primary_particle->p_prng_state_2_[index] ^=
    (p_primary_particle->p_prng_state_2_[index] << 5);
  p_primary_particle->p_prng_state_2_[index] ^=
    (p_primary_particle->p_prng_state_2_[index] >> 7);
  p_primary_particle->p_prng_state_2_[index] ^=
    (p_primary_particle->p_prng_state_2_[index] << 22);
  int t =
    p_primary_particle->p_prng_state_3_[index]
    + p_primary_particle->p_prng_state_4_[index]
    + p_primary_particle->p_prng_state_5_[index];
  p_primary_particle->p_prng_state_3_[index] =
    p_primary_particle->p_prng_state_4_[index];
  p_primary_particle->p_prng_state_5_[index] = t < 0;
  p_primary_particle->p_prng_state_4_[index] = t & 2147483647;
  p_primary_particle->p_prng_state_1_[index] += 1411392427;

  return ((float)(p_primary_particle->p_prng_state_1_[index]
    + p_primary_particle->p_prng_state_2_[index]
    + p_primary_particle->p_prng_state_4_[index])
    //  UINT_MAX       1.0  - float32_precision
    / 4294967295.0) * (1.0f - 1.0f/(1<<23));
}

//inline unsigned int prng_poisson(ParticlesData *particles, unsigned int id, float lambda )
//{
//  ;
//}

//inline float prng_gaussian(ParticlesData *particles, unsigned int id, float sigma)
//{
//  ;
//}

#endif // End of GUARD_GGEMS_GLOBAL_OPENCL_MANAGER_HH
