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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline float prng_poisson(__global PrimaryParticles* p_primary_particle, int const index, float const mean)
  \param p_primary_particle - pointer on primary particles on device
  \param index - index of thread
  \param mean - mean of the Poisson distribution
  \brief Poisson random from G4Poisson in Geant4
*/
inline unsigned int prng_poisson(__global PrimaryParticles* p_primary_particle,
  int const index, float const mean)
{
  // Initialization of parameters
  unsigned int number = 0;
  float position, poisson_value, poisson_sum;
  float value, y, t;

  if (mean <= 16.) {// border == 16, gaussian after 16
    // to avoid 1 due to f32 approximation
    do {
      position = prng_uniform(p_primary_particle, index);
    }
    while ((1.f-position) < 2.e-7f);

    poisson_value = exp(-mean);
    poisson_sum = poisson_value;
    //  v---- Why ? It's not in G4Poisson - JB
    while ((poisson_sum <= position) && (number < 40000.)) {
      number++;
      poisson_value *= mean/number;
      // Not in G4, is it to manage f32 ?  - JB
      if ((poisson_sum + poisson_value) == poisson_sum) break;
      poisson_sum += poisson_value;
    }
    return number;
  }

  t = sqrt(-2.f*log(prng_uniform(p_primary_particle, index)));
  y = 2.f * PI_GPU * prng_uniform(p_primary_particle, index);
  t *= cos(y);
  value = mean + t*sqrt(mean) + 0.5f;

  if (value <= 0.) return (unsigned int)0;

  return (value >= 2.e9f) ? (unsigned int)2.e9f : (unsigned int)value;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline float prng_gaussian(__global PrimaryParticles* p_primary_particle, int const index, float const sigma)
  \param p_primary_particle - pointer on primary particles on device
  \param index - index of thread
  \param sigma - standard deviation
  \brief Gaussian random
*/
inline float prng_gaussian(__global PrimaryParticles* p_primary_particle,
  int const index, float const sigma)
{
  // Box-Muller transformation
  float const u1 = prng_uniform(p_primary_particle, index);
  float const u2 = prng_uniform(p_primary_particle, index);
  float const r1 = sqrt(-2.0f * log(u1));
  float const r2 = 2.0f * PI_GPU * u2;

  return sigma * r1 * cos(r2);
}

#endif // End of GUARD_GGEMS_GLOBAL_OPENCL_MANAGER_HH
