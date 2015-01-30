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
__host__ __device__ f32xyz fmatrixfxyz_mul(f32matrix33 m, f32xyz u) {
    f32xyz r = {m.a*u.x + m.b*u.y + m.c*u.z,
                m.d*u.x + m.e*u.y + m.f*u.z,
                m.g*u.x + m.h*u.y + m.i*u.z};
    return r;
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

    return fmatrixfxyz_mul(rot, u);
}

// Rotate a vector u around the x-axis
__host__ __device__ f32xyz fxyz_rotate_x_axis(f32xyz u, f32 angle) {
    angle *= deg;
    f32 cs = cos(angle);
    f32 sn = sin(angle);

    f32matrix33 rot = {1.0, 0.0, 0.0,
                       0.0, cs, -sn,
                       0.0, sn, cs};

    return fmatrixfxyz_mul(rot, u);
}

// Rotate a vector u around the y-axis
__host__ __device__ f32xyz fxyz_rotate_y_axis(f32xyz u, f32 angle) {
    angle *= deg;
    f32 cs = cos(angle);
    f32 sn = sin(angle);

    f32matrix33 rot = {cs, 0.0, sn,
                       0.0, 1.0, 0.0,
                       -sn, 0.0, cs};

    return fmatrixfxyz_mul(rot, u);
}

// Rotate a vector u around the z-axis
__host__ __device__ f32xyz fxyz_rotate_z_axis(f32xyz u, f32 angle) {
    angle *= deg;
    f32 cs = cos(angle);
    f32 sn = sin(angle);

    f32matrix33 rot = {cs, -sn, 0.0,
                       sn, cs, 0.0,
                       0.0, 0.0, 1.0};

    return fmatrixfxyz_mul(rot, u);
}

// Return abs
__host__ __device__ f32xyz fxyz_abs(f32xyz u) {
    u.x = fabs(u.x);
    u.y = fabs(u.y);
    u.z = fabs(u.z);
    return u;
}

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



#endif



#endif
