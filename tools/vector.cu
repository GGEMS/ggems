// GGEMS Copyright (C) 2015

/*!
 * \file vector.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef VECTOR_CU
#define VECTOR_CU

#include "vector.cuh"

/// Single precision functions //////////////////////////////////////////////

// r = u - v
__host__ __device__ f32xyz fxyz_sub(f32xyz u, f32xyz v) {
    f32xyz r={u.x-v.x, u.y-v.y, u.z-v.z};
    return r;
}

// r = u + v
__host__ __device__ f32xyz fxyz_add(f32xyz u, f32xyz v) {
    f32xyz r={u.x+v.x, u.y+v.y, u.z+v.z};
    return r;
}

// r = u * v
__host__ __device__ f32xyz fxyz_mul(f32xyz u, f32xyz v) {
    f32xyz r={u.x*v.x, u.y*v.y, u.z*v.z};
    return r;
}

// r = u / v
__host__ __device__ f32xyz fxyz_div(f32xyz u, f32xyz v) {
    f32xyz r={u.x/v.x, u.y/v.y, u.z/v.z};
    return r;
}

// r = u * s
__host__ __device__ f32xyz fxyz_scale(f32xyz u, f32 s) {
    f32xyz r={u.x*s, u.y*s, u.z*s};
    return r;
}

// r = u . v
__host__ __device__ f32 fxyz_dot(f32xyz u, f32xyz v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

// r = u x v
__host__ __device__ f32xyz fxyz_cross(f32xyz u, f32xyz v) {
    f32xyz r;
    r.x = u.y*v.z - u.z*v.y;
    r.y = u.z*v.x - u.x*v.z;
    r.z = u.x*v.y - u.y*v.x;
    return r;
}

// r = m * u
__host__ __device__ f32xyz fmatrix_mul_fxyz(f32matrix33 m, f32xyz u) {
    f32xyz r = {m.m00*u.x + m.m01*u.y + m.m02*u.z,
                m.m10*u.x + m.m11*u.y + m.m12*u.z,
                m.m20*u.x + m.m21*u.y + m.m22*u.z};
    return r;
}
__host__ __device__ f32xyz fmatrix_mul_fxyz(f32matrix44 m, f32xyz u) {
    f32xyz r = { m.m00*u.x + m.m01*u.y + m.m02*u.z + m.m03*1.0f,
                 m.m10*u.x + m.m11*u.y + m.m12*u.z + m.m13*1.0f,
                 m.m20*u.x + m.m21*u.y + m.m22*u.z + m.m23*1.0f };
    return r;
}

// r = m * n
__host__ __device__ f32matrix44 fmatrix_mul( f32matrix44 m, f32matrix44 n )
{
    f32matrix44 res;

    // first row
    res.m00 = m.m00*n.m00 + m.m01*n.m10 + m.m02*n.m20 + m.m03*n.m30;
    res.m01 = m.m00*n.m01 + m.m01*n.m11 + m.m02*n.m21 + m.m03*n.m31;
    res.m02 = m.m00*n.m02 + m.m01*n.m12 + m.m02*n.m22 + m.m03*n.m32;
    res.m03 = m.m00*n.m03 + m.m01*n.m13 + m.m02*n.m23 + m.m03*n.m33;

    // second row
    res.m10 = m.m10*n.m00 + m.m11*n.m10 + m.m12*n.m20 + m.m13*n.m30;
    res.m11 = m.m10*n.m01 + m.m11*n.m11 + m.m12*n.m21 + m.m13*n.m31;
    res.m12 = m.m10*n.m02 + m.m11*n.m12 + m.m12*n.m22 + m.m13*n.m32;
    res.m13 = m.m10*n.m03 + m.m11*n.m13 + m.m12*n.m23 + m.m13*n.m33;

    // third row
    res.m20 = m.m20*n.m00 + m.m21*n.m10 + m.m22*n.m20 + m.m23*n.m30;
    res.m21 = m.m20*n.m01 + m.m21*n.m11 + m.m22*n.m21 + m.m23*n.m31;
    res.m22 = m.m20*n.m02 + m.m21*n.m12 + m.m22*n.m22 + m.m23*n.m32;
    res.m23 = m.m20*n.m03 + m.m21*n.m13 + m.m22*n.m23 + m.m23*n.m33;

    // row #4
    res.m30 = m.m30*n.m00 + m.m31*n.m10 + m.m32*n.m20 + m.m33*n.m30;
    res.m31 = m.m30*n.m01 + m.m31*n.m11 + m.m32*n.m21 + m.m33*n.m31;
    res.m32 = m.m30*n.m02 + m.m31*n.m12 + m.m32*n.m22 + m.m33*n.m32;
    res.m33 = m.m30*n.m03 + m.m31*n.m13 + m.m32*n.m23 + m.m33*n.m33;

    return res;
}

// r = m^T
__host__ __device__ f32matrix44 fmatrix_trans( f32matrix44 m )
{
    f32 tmp;

    tmp = m.m01; m.m01 = m.m10; m.m10 = tmp;
    tmp = m.m02; m.m02 = m.m20; m.m20 = tmp;
    tmp = m.m03; m.m03 = m.m30; m.m30 = tmp;
    tmp = m.m12; m.m12 = m.m21; m.m21 = tmp;
    tmp = m.m13; m.m13 = m.m31; m.m31 = tmp;
    tmp = m.m23; m.m23 = m.m32; m.m32 = tmp;

    return m;
}

// return an unitary vector
__host__ __device__ f32xyz fxyz_unit(f32xyz u) {
    f32 imag = 1.0f / sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
    return make_f32xyz(u.x*imag, u.y*imag, u.z*imag);
}

// rotate a vector u (Euler)
__host__ __device__ f32xyz fxyz_rotate_euler(f32xyz u, f32xyz EulerAngles) {

    f32 phi = EulerAngles.x*deg; // deg is defined by G4 unit system
    f32 theta = EulerAngles.y*deg;
    f32 psi = EulerAngles.z*deg;

    f32 sph = sin(phi);
    f32 cph = cos(phi);
    f32 sth = sin(theta);
    f32 cth = cos(theta);
    f32 sps = sin(psi);
    f32 cps = cos(psi);

    // Build rotation matrix
    f32matrix33 rot = { cph*cps-sph*cth*sps,  cph*sps+sph*cth*cps,  sth*sph,
                       -sph*cps-cph*cth*sps, -sph*sps+cph*cth*cps,  sth*cph,
                        sth*sps,             -sth*cps,                  cth};

    return fmatrix_mul_fxyz(rot, u);
}

// Rotate a vector u around the x-axis
__host__ __device__ f32xyz fxyz_rotate_x_axis(f32xyz u, f32 angle) {
    //angle *= deg;
    f32 cs = cos(angle);
    f32 sn = sin(angle);

    f32matrix33 rot = {1.0, 0.0, 0.0,
                       0.0, cs, -sn,
                       0.0, sn, cs};

    return fmatrix_mul_fxyz(rot, u);
}

// Rotate a vector u around the y-axis
__host__ __device__ f32xyz fxyz_rotate_y_axis(f32xyz u, f32 angle) {
    //angle *= deg;

    f32 cs = cos(angle);
    f32 sn = sin(angle);

    f32matrix33 rot = {cs, 0.0, sn,
                       0.0, 1.0, 0.0,
                       -sn, 0.0, cs};

    return fmatrix_mul_fxyz(rot, u);
}

// Rotate a vector u around the z-axis
__host__ __device__ f32xyz fxyz_rotate_z_axis(f32xyz u, f32 angle) {
    //angle *= deg;
    f32 cs = cos(angle);
    f32 sn = sin(angle);

    f32matrix33 rot = {cs, -sn, 0.0,
                       sn, cs, 0.0,
                       0.0, 0.0, 1.0};

    return fmatrix_mul_fxyz(rot, u);
}

// Return abs
__host__ __device__ f32xyz fxyz_abs(f32xyz u) {
    u.x = fabs(u.x);
    u.y = fabs(u.y);
    u.z = fabs(u.z);
    return u;
}

// Inverse value of a vector 1/u
__host__ __device__ f32xyz fxyz_inv(f32xyz u) {
    u.x = 1.0 / u.x;
    u.y = 1.0 / u.y;
    u.z = 1.0 / u.z;
    return u;
}


/// Transform calculator ////////////////////////////////////////////////////

// Convert a point from local to global frame
__host__ __device__ f32xyz fxyz_local_to_global_frame( f32matrix44 G, f32xyz u )
{
    return fmatrix_mul_fxyz( G, u );
}

// Convert a point from global to local frame
__host__ __device__ f32xyz fxyz_global_to_local_frame( f32matrix44 G, f32xyz u)
{
    // first transpose the transformation matrix to apply an inverse transform
    return fmatrix_mul_fxyz( fmatrix_trans( G ), u );
}

TransformCalculator::TransformCalculator()
{
    // Init identity transformation
    m_T = make_f32matrix44( 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1 );

    m_R = make_f32matrix44( 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1 );

    m_P0 = make_f32matrix44( 1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1 );
}


void TransformCalculator::set_translation( f32 tx, f32 ty, f32 tz )
{
    m_T = make_f32matrix44( 1, 0, 0, tx,
                            0, 1, 0, ty,
                            0, 0, 1, tz,
                            0, 0, 0,  1 );
}

void TransformCalculator::set_rotation( f32 rx, f32 ry, f32 rz )
{
    // Rotation convention xyz
    f32 cs, sn;

    // x
    cs = cos( rx );
    sn = sin( rx );

    f32matrix44 Rx = { 1.0, 0.0, 0.0, 0.0,
                       0.0,  cs, -sn, 0.0,
                       0.0,  sn,  cs, 0.0,
                       0.0, 0.0, 0.0, 1.0 };

    // y
    cs = cos( ry );
    sn = sin( ry );

    f32matrix44 Ry = {  cs, 0.0,  sn, 0.0,
                       0.0, 1.0, 0.0, 0.0,
                       -sn, 0.0,  cs, 0.0,
                       0.0, 0.0, 0.0, 1.0 };

    // z
    cs = cos( rz );
    sn = sin( rz );

    f32matrix44 Rz = { cs,  -sn, 0.0, 0.0,
                       sn,   cs, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 1.0 };

    // Get R (Rz*Ry*Rx)
    m_R = fmatrix_mul( Ry, Rx );
    m_R = fmatrix_mul( Rz, m_R );
}

void TransformCalculator::set_rotation( f32xyz angles )
{
    set_rotation( angles.x, angles.y, angles.z );
}

void TransformCalculator::set_axis_transformation( f32matrix33 P )
{
    m_P0 = make_f32matrix44( P.m00, P.m01, P.m02, 0.0,
                             P.m10, P.m11, P.m12, 0.0,
                             P.m20, P.m21, P.m22, 0.0,
                               0.0,   0.0,   0.0, 1.0 );
}

f32matrix44 TransformCalculator::get_translation_matrix()
{
    return m_T;
}

f32matrix44 TransformCalculator::get_rotation_matrix()
{
    return m_R;
}

f32matrix44 TransformCalculator::get_projection_matrix()
{
    return m_P0;
}

f32matrix44 TransformCalculator::get_transformation_matrix()
{
    // compute the matrix G = R*T*P0
    return fmatrix_mul( m_R, fmatrix_mul( m_T, m_P0 ) );
}

/*

/// Double precision functions ///////////////////////////////////////////////

#ifndef SINGLE_PRECISION
    // Add function with double precision


// r = u - v
__host__ __device__ f64xyz fxyz_sub(f64xyz u, f64xyz v) {
    f64xyz r={u.x-v.x, u.y-v.y, u.z-v.z};
    return r;
}

// r = u + v
__host__ __device__ f64xyz fxyz_add(f64xyz u, f64xyz v) {
    f64xyz r={u.x+v.x, u.y+v.y, u.z+v.z};
    return r;
}

// r = u * v
__host__ __device__ f64xyz fxyz_mul(f64xyz u, f64xyz v) {
    f64xyz r={u.x*v.x, u.y*v.y, u.z*v.z};
    return r;
}

// r = u / v
__host__ __device__ f64xyz fxyz_div(f64xyz u, f64xyz v) {
    f64xyz r={u.x/v.x, u.y/v.y, u.z/v.z};
    return r;
}

// r = u * s
__host__ __device__ f64xyz fxyz_scale(f64xyz u, f64 s) {
    f64xyz r={u.x*s, u.y*s, u.z*s};
    return r;
}

// r = u . v
__host__ __device__ f64 fxyz_dot(f64xyz u, f64xyz v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

// r = u x v
__host__ __device__ f64xyz fxyz_cross(f64xyz u, f64xyz v) {
    f64xyz r;
    r.x = u.y*v.z - u.z*v.y;
    r.y = u.z*v.x - u.x*v.z;
    r.z = u.x*v.y - u.y*v.x;
    return r;
}

// r = m * u
__host__ __device__ f64xyz fmatrixfxyz_mul(f64matrix33 m, f64xyz u) {
    f64xyz r = {m.a*u.x + m.b*u.y + m.c*u.z,
                m.d*u.x + m.e*u.y + m.f*u.z,
                m.g*u.x + m.h*u.y + m.i*u.z};
    return r;
}

// return an unitary vector
__host__ __device__ f64xyz fxyz_unit(f64xyz u) {
    f64 imag = 1.0f / sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
    return make_f64xyz(u.x*imag, u.y*imag, u.z*imag);
}

// rotate a vector u (Euler)
__host__ __device__ f64xyz fxyz_rotate_euler(f64xyz u, f64xyz EulerAngles) {

    f64 phi = EulerAngles.x*deg; // deg is defined by G4 unit system
    f64 theta = EulerAngles.y*deg;
    f64 psi = EulerAngles.z*deg;

    f64 sph = sin(phi);
    f64 cph = cos(phi);
    f64 sth = sin(theta);
    f64 cth = cos(theta);
    f64 sps = sin(psi);
    f64 cps = cos(psi);

    // Build rotation matrix
    f64matrix33 rot = { cph*cps-sph*cth*sps,  cph*sps+sph*cth*cps,  sth*sph,
                       -sph*cps-cph*cth*sps, -sph*sps+cph*cth*cps,  sth*cph,
                        sth*sps,             -sth*cps,                  cth};

    return fmatrixfxyz_mul(rot, u);
}

// Rotate a vector u around the x-axis
__host__ __device__ f64xyz fxyz_rotate_x_axis(f64xyz u, f64 angle) {
    angle *= deg;
    f64 cs = cos(angle);
    f64 sn = sin(angle);

    f64matrix33 rot = {1.0, 0.0, 0.0,
                       0.0, cs, -sn,
                       0.0, sn, cs};

    return fmatrixfxyz_mul(rot, u);
}

// Rotate a vector u around the y-axis
__host__ __device__ f64xyz fxyz_rotate_y_axis(f64xyz u, f64 angle) {
    angle *= deg;
    f64 cs = cos(angle);
    f64 sn = sin(angle);

    f64matrix33 rot = {cs, 0.0, sn,
                       0.0, 1.0, 0.0,
                       -sn, 0.0, cs};

    return fmatrixfxyz_mul(rot, u);
}

// Rotate a vector u around the z-axis
__host__ __device__ f64xyz fxyz_rotate_z_axis(f64xyz u, f64 angle) {
    angle *= deg;
    f64 cs = cos(angle);
    f64 sn = sin(angle);

    f64matrix33 rot = {cs, -sn, 0.0,
                       sn, cs, 0.0,
                       0.0, 0.0, 1.0};

    return fmatrixfxyz_mul(rot, u);
}

// Return abs
__host__ __device__ f64xyz fxyz_abs(f64xyz u) {
    u.x = fabs(u.x);
    u.y = fabs(u.y);
    u.z = fabs(u.z);
    return u;
}

// Inverse value of a vector 1/u
__host__ __device__ f64xyz fxyz_inv(f64xyz u) {
    u.x = 1.0 / u.x;
    u.y = 1.0 / u.y;
    u.z = 1.0 / u.z;
    return u;
}

#endif
*/

#endif
