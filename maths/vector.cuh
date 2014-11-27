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

#ifndef VECTOR_H
#define VECTOR_H

/////////////////////////////////////////////////////////////////////////////
// Maths
/////////////////////////////////////////////////////////////////////////////

#include "constants.cuh"
#include "global.cuh"

#ifndef MATRIX3
#define MATRIX3
struct matrix3 {
    f32 a, b, c, d, e, f, g, h, i;
};
#endif

// r = u - v
__host__ __device__ f32xyz f3_sub(f32xyz u, f32xyz v);
// r = u + v
__host__ __device__ f32xyz f3_add(f32xyz u, f32xyz v);
// r = u * v
__host__ __device__ f32xyz f3_mul(f32xyz u, f32xyz v);
// r = u / v
__host__ __device__ f32xyz f3_div(f32xyz u, f32xyz v);
// r = u * s
__host__ __device__ f32xyz f3_scale(f32xyz u, f32 s);
// r = u . v
__host__ __device__ f32 f3_dot(f32xyz u, f32xyz v);
// r = u x v
__host__ __device__ f32xyz f3_cross(f32xyz u, f32xyz v);
// r = m * u
__host__ __device__ f32xyz m3f3_mul(matrix3 matrix, f32xyz u);
// return an unitary vector
__host__ __device__ f32xyz f3_unit(f32xyz u);
// rotate a vector u
__host__ __device__ f32xyz f3_rotate(f32xyz u, f32xyz EulerAngles); // phi, theta, psi

#endif
