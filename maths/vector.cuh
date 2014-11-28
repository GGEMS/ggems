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

#ifndef F32MATRIX33
#define F32MATRIX33
struct f32matrix33 {
    f32 a, b, c, d, e, f, g, h, i;
};
#endif

#ifndef F64MATRIX33
#define F64MATRIX33
struct f64matrix33 {
    f64 a, b, c, d, e, f, g, h, i;
};
#endif

// r = u - v
__host__ __device__ f32xyz fxyz_sub(f32xyz u, f32xyz v);
__host__ __device__ f64xyz fxyz_sub(f64xyz u, f64xyz v);
// r = u + v
__host__ __device__ f32xyz fxyz_add(f32xyz u, f32xyz v);
__host__ __device__ f64xyz fxyz_add(f64xyz u, f64xyz v);
// r = u * v
__host__ __device__ f32xyz fxyz_mul(f32xyz u, f32xyz v);
__host__ __device__ f64xyz fxyz_mul(f64xyz u, f64xyz v);
// r = u / v
__host__ __device__ f32xyz fxyz_div(f32xyz u, f32xyz v);
__host__ __device__ f64xyz fxyz_div(f64xyz u, f64xyz v);
// r = u * s
__host__ __device__ f32xyz fxyz_scale(f32xyz u, f32 s);
__host__ __device__ f64xyz fxyz_scale(f64xyz u, f64 s);
// r = u . v
__host__ __device__ f32 fxyz_dot(f32xyz u, f32xyz v);
__host__ __device__ f64 fxyz_dot(f64xyz u, f64xyz v);
// r = u x v
__host__ __device__ f32xyz fxyz_cross(f32xyz u, f32xyz v);
__host__ __device__ f64xyz fxyz_cross(f64xyz u, f64xyz v);
// r = m * u
__host__ __device__ f32xyz fmatrixfxyz_mul(f32matrix33 matrix, f32xyz u);
__host__ __device__ f64xyz fmatrixfxyz_mul(f64matrix33 matrix, f64xyz u);
// return an unitary vector
__host__ __device__ f32xyz fxyz_unit(f32xyz u);
__host__ __device__ f64xyz fxyz_unit(f64xyz u);
// rotate a vector u
__host__ __device__ f32xyz fxyz_rotate(f32xyz u, f32xyz EulerAngles); // phi, theta, psi
__host__ __device__ f64xyz fxyz_rotate(f64xyz u, f64xyz EulerAngles); // phi, theta, psi
#endif
