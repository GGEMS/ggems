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

#ifndef RAYTRACING_H
#define RAYTRACING_H

#include "vector.cuh"
#include "global.cuh"
#include "constants.cuh"


// Overlap test return (short int):
//       -1 No interection         +1 Intersection

// Hit collision return (f32):
//        t >= 0 Distance of collision (if no collision t = FLT_MAX)


// AABB/Triangle test - Akenine-Moller algorithm
__host__ __device__ short int overlap_AABB_triangle(f32 xmin, f32 xmax,        // AABB
                                                    f32 ymin, f32 ymax,
                                                    f32 zmin, f32 zmax,
                                                    float3 u, float3 v, float3 w); // Triangle
// Ray/Sphere intersection
__host__ __device__ f32 hit_ray_sphere(float3 ray_p, float3 ray_d,           // Ray
                                         float3 sphere_c, f32 sphere_rad);  // Sphere


// Ray/AABB intersection - Smits algorithm
__host__ __device__ f32 hit_ray_AABB(float3 ray_p, float3 ray_d,
                                       f32 aabb_xmin, f32 aabb_xmax,
                                       f32 aabb_ymin, f32 aabb_ymax,
                                       f32 aabb_zmin, f32 aabb_zmax);

// Ray/AABB test - Smits algorithm
__host__ __device__ bool test_ray_AABB(float3 ray_p, float3 ray_d,
                                       f32 aabb_xmin, f32 aabb_xmax,
                                       f32 aabb_ymin, f32 aabb_ymax,
                                       f32 aabb_zmin, f32 aabb_zmax);

// AABB/AABB test
__host__ __device__ bool test_AABB_AABB(f32 a_xmin, f32 a_xmax, f32 a_ymin, f32 a_ymax,
                                        f32 a_zmin, f32 a_zmax,
                                        f32 b_xmin, f32 b_xmax, f32 b_ymin, f32 b_ymax,
                                        f32 b_zmin, f32 b_zmax);

// Point/AABB test
__host__ __device__ bool test_point_AABB(float3 p,
                                         f32 aabb_xmin, f32 aabb_xmax,
                                         f32 aabb_ymin, f32 aabb_ymax,
                                         f32 aabb_zmin, f32 aabb_zmax);

// Ray/triangle intersection - Moller-Trumbore algorithm
__host__ __device__ f32 hit_ray_triangle(float3 ray_p, float3 ray_d,
                                           float3 tri_u,              // Triangle
                                           float3 tri_v,
                                           float3 tri_w);

#endif
