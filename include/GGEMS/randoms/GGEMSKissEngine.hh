#ifndef GUARD_GGEMS_RANDOMS_GGEMSKISSENGINE_HH
#define GUARD_GGEMS_RANDOMS_GGEMSKISSENGINE_HH

/*!
  \file GGEMSKissEngine.hh

  \brief Functions for pseudo random number generator using KISS (Keep it Simple Stupid) engine. This functions can be used only by an OpenCL kernel

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#ifdef OPENCL_COMPILER

#include "GGEMS/randoms/GGEMSRandomStack.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline float kiss_uniform(__global Random* p_random, int const index)
  \param p_random - pointer on random buffer on OpenCL device
  \param index - index of thread
  \brief JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder
  and the BigCrunch tests in TestU01
*/
inline float kiss_uniform(__global Random* p_random, GGint const index)
{
  // y ^= (y<<5);
  // y ^= (y>>7);
  // y ^= (y<<22);
  // t = z+w+c;
  // z = w;
  // c = t < 0;
  // w = t & 2147483647;
  // x += 1411392427;

  p_random->p_prng_state_2_[index] ^= p_random->p_prng_state_2_[index] << 5;
  p_random->p_prng_state_2_[index] ^= p_random->p_prng_state_2_[index] >> 7;
  p_random->p_prng_state_2_[index] ^= p_random->p_prng_state_2_[index] << 22;

  GGint t = p_random->p_prng_state_3_[index] + p_random->p_prng_state_4_[index]
    + p_random->p_prng_state_5_[index];

  p_random->p_prng_state_3_[index] = p_random->p_prng_state_4_[index];
  p_random->p_prng_state_5_[index] = t < 0;
  p_random->p_prng_state_4_[index] = t & 2147483647;
  p_random->p_prng_state_1_[index] += 1411392427;

  return ((float)(p_random->p_prng_state_1_[index]
    + p_random->p_prng_state_2_[index]
    + p_random->p_prng_state_4_[index])
    //  UINT_MAX       1.0  - float32_precision
    / 4294967295.0) * (1.0f - 1.0f/(1<<23));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn unsigned int kiss_poisson(__global Random* p_random, int const index, float const mean)
  \param p_random - pointer on random buffer on OpenCL device
  \param index - index of thread
  \param mean - mean of the Poisson distribution
  \brief Poisson random from G4Poisson in Geant4
*/
inline unsigned int kiss_poisson(__global Random* p_random, GGint const index,
  GGfloat const mean)
{
  // Initialization of parameters
  GGuint number = 0;
  GGfloat position = 0.0, poisson_value = 0.0, poisson_sum = 0.0;
  GGfloat value = 0.0, y = 0.0, t = 0.0;

  if (mean <= 16.) {// border == 16, gaussian after 16
    // to avoid 1 due to f32 approximation
    do {
      position = kiss_uniform(p_random, index);
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

  t = sqrt(-2.f*log(kiss_uniform(p_random, index)));
  y = 2.f * PI * kiss_uniform(p_random, index);
  t *= cos(y);
  value = mean + t*sqrt(mean) + 0.5f;

  if (value <= 0.) return (unsigned int)0;

  return (value >= 2.e9f) ? (unsigned int)2.e9f : (unsigned int)value;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline float kiss_gauss(__global Random* p_random, GGint const index, GGfloat const sigma)
  \param p_random - pointer on random buffer on OpenCL device
  \param index - index of thread
  \param sigma - standard deviation
  \brief Gaussian random
*/
inline float kiss_gauss(__global Random* p_random, GGint const index,
  GGfloat const sigma)
{
  // Box-Muller transformation
  GGfloat const u1 = kiss_uniform(p_random, index);
  GGfloat const u2 = kiss_uniform(p_random, index);
  GGfloat const r1 = sqrt(-2.0f * log(u1));
  GGfloat const r2 = 2.0f * PI * u2;

  return sigma * r1 * cos(r2);
}

#endif

#endif // End of GUARD_GGEMS_RANDOMS_GGEMSKISSENGINE_HH
