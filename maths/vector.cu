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

// r = u - v
__host__ __device__ float3 f3_sub(float3 u, float3 v) {
    float3 r={u.x-v.x, u.y-v.y, u.z-v.z};
    return r;
}

// r = u + v
__host__ __device__ float3 f3_add(float3 u, float3 v) {
    float3 r={u.x+v.x, u.y+v.y, u.z+v.z};
    return r;
}

// r = u * v
__host__ __device__ float3 f3_mul(float3 u, float3 v) {
    float3 r={u.x*v.x, u.y*v.y, u.z*v.z};
    return r;
}

// r = u / v
__host__ __device__ float3 f3_div(float3 u, float3 v) {
    float3 r={u.x/v.x, u.y/v.y, u.z/v.z};
    return r;
}

// r = u * s
__host__ __device__ float3 f3_scale(float3 u, f32 s) {
    float3 r={u.x*s, u.y*s, u.z*s};
    return r;
}

// r = u . v
__host__ __device__ f32 f3_dot(float3 u, float3 v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

// r = u x v
__host__ __device__ float3 f3_cross(float3 u, float3 v) {
    float3 r;
    r.x = u.y*v.z - u.z*v.y;
    r.y = u.z*v.x - u.x*v.z;
    r.z = u.x*v.y - u.y*v.x;
    return r;
}

// r = m * u
__host__ __device__ float3 m3f3_mul(matrix3 m, float3 u) {
    float3 r = {m.a*u.x + m.b*u.y + m.c*u.z,
                m.d*u.x + m.e*u.y + m.f*u.z,
                m.g*u.x + m.h*u.y + m.i*u.z};
    return r;
}

// return an unitary vector
__host__ __device__ float3 f3_unit(float3 u) {
    f32 imag = 1.0f / sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
    return make_float3(u.x*imag, u.y*imag, u.z*imag);
}

// rotate a vector u
__host__ __device__ float3 f3_rotate(float3 u, float3 EulerAngles) {

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
    matrix3 rot = { cph*cps-sph*cth*sps,  cph*sps+sph*cth*cps,  sth*sph,
                    -sph*cps-cph*cth*sps, -sph*sps+cph*cth*cps,  sth*cph,
                    sth*sps,             -sth*cps,                  cth};

    return m3f3_mul(rot, u);
}

#endif
