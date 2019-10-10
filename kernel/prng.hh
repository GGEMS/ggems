#ifndef GUARD_GGEMS_KERNEL_PRNG_HH
#define GUARD_GGEMS_KERNEL_PRNG_HH

/*!
  \file prng.hh

  \brief Function for pseudo random number generator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday October 10, 2019
*/

/*!
  \fn
  \param
  \brief JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder
  and the BigCrunch tests in TestU01
*/
float prng_uniform(__global PrimaryParticles* p_primary_particle)
{
  // y ^= (y<<5);
  // y ^= (y>>7);
  // y ^= (y<<22);
  // t = z+w+c;
  // z = w;
  // c = t < 0;
  // w = t & 2147483647;
  // x += 1411392427;

  *p_primary_particle->p_prng_state_2_ ^=
    (*p_primary_particle->p_prng_state_2_ << 5);
  *p_primary_particle->p_prng_state_2_ ^=
    (*p_primary_particle->p_prng_state_2_ >> 7);
  *p_primary_particle->p_prng_state_2_ ^=
    (*p_primary_particle->p_prng_state_2_ << 22);
  int t = *p_primary_particle->p_prng_state_3_
    + *p_primary_particle->p_prng_state_4_
    + *p_primary_particle->p_prng_state_5_;
  *p_primary_particle->p_prng_state_3_ = *p_primary_particle->p_prng_state_4_;
  *p_primary_particle->p_prng_state_5_ = t < 0;
  *p_primary_particle->p_prng_state_4_ = t & 2147483647;
  *p_primary_particle->p_prng_state_1_ += 1411392427;

  // For the f32 version is more tricky
  float rndm_value = ((float)(*p_primary_particle->p_prng_state_1_
    + *p_primary_particle->p_prng_state_2_
    + *p_primary_particle->p_prng_state_4_) / 4294967295.0)
    * (1.0f - 1.0f/(1<<23));

  return rndm_value;
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
