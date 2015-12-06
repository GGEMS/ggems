// GGEMS Copyright (C) 2015

/*!
 * \file prng.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef PRNG_H
#define PRNG_H

#include "global.cuh"
#include "particles.cuh"

/////////////////////////////////////////////////////////////////////////////
// Prng
/////////////////////////////////////////////////////////////////////////////

__host__  __device__ f32 JKISS32(ParticlesData &particles, ui32 id);

#endif
