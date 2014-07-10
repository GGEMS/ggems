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

#ifndef MATRIX3
#define MATRIX3
struct matrix3 {
    float a, b, c, d, e, f, g, h, i;
};
#endif

// r = u - v
__host__ __device__ float3 f3_sub(float3 u, float3 v);
// r = u + v
__host__ __device__ float3 f3_add(float3 u, float3 v);
// r = u * v
__host__ __device__ float3 f3_mul(float3 u, float3 v);
// r = u / v
__host__ __device__ float3 f3_div(float3 u, float3 v);
// r = u * s
__host__ __device__ float3 f3_scale(float3 u, float s);
// r = u . v
__host__ __device__ float f3_dot(float3 u, float3 v);
// r = u x v
__host__ __device__ float3 f3_cross(float3 u, float3 v);
// r = m * u
__host__ __device__ float3 m3f3_mul(matrix3 m, float3 u);
// return an unitary vector
__host__ __device__ float3 f3_unit(float3 u);
}
#endif
