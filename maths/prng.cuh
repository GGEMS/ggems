// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef PRNG_H
#define PRNG_H

#include "global.cuh"
#include "particles.cuh"

/////////////////////////////////////////////////////////////////////////////
// Prng
/////////////////////////////////////////////////////////////////////////////

__host__  __device__ f32 JKISS32(ParticleStack &particles, ui32 id);
//__device__ unsigned long brent_int(ui32 index, unsigned long *device_x_brent, unsigned long seed);
//__device__ double Brent_real(int index, unsigned long *device_x_brent, unsigned long seed);

#endif
