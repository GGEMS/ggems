#ifndef GUARD_GGEMS_RANDOMS_GGEMSKISSENGINE_HH
#define GUARD_GGEMS_RANDOMS_GGEMSKISSENGINE_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSKissEngine.hh

  \brief Functions for pseudo random number generator using KISS (Keep it Simple Stupid) engine. This functions can be used only by an OpenCL kernel

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#ifdef __OPENCL_C_VERSION__

#include "GGEMS/randoms/GGEMSRandom.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat KissUniform(global GGEMSRandom* random, GGint const index)
  \param random - pointer on random buffer on OpenCL device
  \param index - index of thread
  \return Uniform random float number
  \brief JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder and the BigCrunch tests in TestU01
*/
inline GGfloat KissUniform(global GGEMSRandom* random, GGint const index)
{
  // y ^= (y<<5);
  // y ^= (y>>7);
  // y ^= (y<<22);
  // t = z+w+c;
  // z = w;
  // c = t < 0;
  // w = t & 2147483647;
  // x += 1411392427;

  random->prng_state_2_[index] ^= random->prng_state_2_[index] << 5;
  random->prng_state_2_[index] ^= random->prng_state_2_[index] >> 7;
  random->prng_state_2_[index] ^= random->prng_state_2_[index] << 22;

  GGint t = random->prng_state_3_[index] + random->prng_state_4_[index] + random->prng_state_5_[index];

  random->prng_state_3_[index] = random->prng_state_4_[index];
  random->prng_state_5_[index] = t < 0;
  random->prng_state_4_[index] = t & 2147483647;
  random->prng_state_1_[index] += 1411392427;

  return ((GGfloat)(random->prng_state_1_[index] + random->prng_state_2_[index] + random->prng_state_4_[index])
    //  UINT_MAX       1.0  - float32_precision
    / 4294967295.0) * (1.0f - 1.0f/(1<<23));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn GGuint KissPoisson(global GGEMSRandom* random, GGint const index, GGfloat const mean)
  \param random - pointer on random buffer on OpenCL device
  \param index - index of thread
  \param mean - mean of the Poisson distribution
  \return Poisson random uint number
  \brief Poisson random from G4Poisson in Geant4
*/
inline GGint KissPoisson(global GGEMSRandom* random, GGint const index, GGfloat const mean)
{
  // Initialization of parameters
  GGint number = 0;
  GGfloat position = 0.0, poisson_value = 0.0, poisson_sum = 0.0;
  GGfloat value = 0.0, y = 0.0, t = 0.0;

  if (mean <= 16.) {// border == 16, gaussian after 16
    // to avoid 1 due to f32 approximation
    do {
      position = KissUniform(random, index);
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

  t = sqrt(-2.f*log(KissUniform(random, index)));
  y = 2.f * PI * KissUniform(random, index);
  t *= cos(y);
  value = mean + t*sqrt(mean) + 0.5f;

  if (value <= 0.) return (GGuint)0;

  return (value >= 2.e9f) ? (GGint)2.e9f : (GGint)value;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat KissGauss(global GGEMSRandom* random, GGint const index, GGfloat const sigma)
  \param random - pointer on random buffer on OpenCL device
  \param index - index of thread
  \param sigma - standard deviation
  \return Gaussian random float number
  \brief Gaussian random
*/
inline GGfloat KissGauss(global GGEMSRandom* random, GGint const index, GGfloat const sigma)
{
  // Box-Muller transformation
  GGfloat u1 = KissUniform(random, index);
  GGfloat u2 = KissUniform(random, index);
  GGfloat r1 = sqrt(-2.0f * log(u1));
  GGfloat r2 = 2.0f * PI * u2;

  return sigma * r1 * cos(r2);
}

#endif

#endif // End of GUARD_GGEMS_RANDOMS_GGEMSKISSENGINE_HH
