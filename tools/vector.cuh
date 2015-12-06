// GGEMS Copyright (C) 2015

/*!
 * \file vector.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef VECTOR_H
#define VECTOR_H

/////////////////////////////////////////////////////////////////////////////
// Maths
/////////////////////////////////////////////////////////////////////////////

#include "global.cuh"

/// Single precision functions //////////////////////////////////////////////

#ifndef F32MATRIX33
#define F32MATRIX33
struct f32matrix33 {
    f32 a, b, c, d, e, f, g, h, i;
};
#endif

// r = u - v
__host__ __device__ f32xyz fxyz_sub(f32xyz u, f32xyz v);
// r = u + v
__host__ __device__ f32xyz fxyz_add(f32xyz u, f32xyz v);
// r = u * v
__host__ __device__ f32xyz fxyz_mul(f32xyz u, f32xyz v);
// r = u / v
__host__ __device__ f32xyz fxyz_div(f32xyz u, f32xyz v);
// r = u * s
__host__ __device__ f32xyz fxyz_scale(f32xyz u, f32 s);
// r = u . v
__host__ __device__ f32 fxyz_dot(f32xyz u, f32xyz v);
// r = u x v
__host__ __device__ f32xyz fxyz_cross(f32xyz u, f32xyz v);
// r = m * u
__host__ __device__ f32xyz fmatrixfxyz_mul(f32matrix33 matrix, f32xyz u);
// return an unitary vector
__host__ __device__ f32xyz fxyz_unit(f32xyz u);
// rotate a vector u
__host__ __device__ f32xyz fxyz_rotate_euler(f32xyz u, f32xyz EulerAngles); // phi, theta, psi
// rotate a vector u around the x-axis
__host__ __device__ f32xyz fxyz_rotate_x_axis(f32xyz u, f32 angle);
// rotate a vector u around the x-axis
__host__ __device__ f32xyz fxyz_rotate_x_axis(f32xyz u, f32 angle);
// rotate a vector u around the y-axis
__host__ __device__ f32xyz fxyz_rotate_y_axis(f32xyz u, f32 angle);
// rotate a vector u around the z-axis
__host__ __device__ f32xyz fxyz_rotate_z_axis(f32xyz u, f32 angle);
// return abs
__host__ __device__ f32xyz fxyz_abs(f32xyz u);
// return 1/u
__host__ __device__ f32xyz fxyz_inv(f32xyz u);

/// Double precision functions ///////////////////////////////////////////////

#ifndef SINGLE_PRECISION
    // Add function with double precision

#ifndef F64MATRIX33
#define F64MATRIX33
struct f64matrix33 {
    f64 a, b, c, d, e, f, g, h, i;
};
#endif

// r = u - v
__host__ __device__ f64xyz fxyz_sub(f64xyz u, f64xyz v);
// r = u + v
__host__ __device__ f64xyz fxyz_add(f64xyz u, f64xyz v);
// r = u * v
__host__ __device__ f64xyz fxyz_mul(f64xyz u, f64xyz v);
// r = u / v
__host__ __device__ f64xyz fxyz_div(f64xyz u, f64xyz v);
// r = u * s
__host__ __device__ f64xyz fxyz_scale(f64xyz u, f64 s);
// r = u . v
__host__ __device__ f64 fxyz_dot(f64xyz u, f64xyz v);
// r = u x v
__host__ __device__ f64xyz fxyz_cross(f64xyz u, f64xyz v);
// r = m * u
__host__ __device__ f64xyz fmatrixfxyz_mul(f64matrix33 matrix, f64xyz u);
// return an unitary vector
__host__ __device__ f64xyz fxyz_unit(f64xyz u);
// rotate a vector u
__host__ __device__ f64xyz fxyz_rotate_euler(f64xyz u, f64xyz EulerAngles); // phi, theta, psi
// rotate a vector u around the x-axis
__host__ __device__ f64xyz fxyz_rotate_x_axis(f64xyz u, f64 angle);
// rotate a vector u around the x-axis
__host__ __device__ f64xyz fxyz_rotate_x_axis(f64xyz u, f64 angle);
// rotate a vector u around the y-axis
__host__ __device__ f64xyz fxyz_rotate_y_axis(f64xyz u, f64 angle);
// rotate a vector u around the z-axis
__host__ __device__ f64xyz fxyz_rotate_z_axis(f64xyz u, f64 angle);
// return abs
__host__ __device__ f64xyz fxyz_abs(f64xyz u);
// return 1/u
__host__ __device__ f64xyz fxyz_inv(f64xyz u);

#endif

#endif
