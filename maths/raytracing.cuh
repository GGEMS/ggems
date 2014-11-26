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
#include <cfloat>
#include <stdlib.h>
#include <stdio.h>
#include "constants.cuh"


// Overlap test return (short int):
//       -1 No interection         +1 Intersection

// Hit collision return (float):
//        t >= 0 Distance of collision (if no collision t = FLT_MAX)


// AABB/Triangle test - Akenine-Moller algorithm
__host__ __device__ short int overlap_AABB_triangle(float xmin, float xmax,        // AABB
                                                    float ymin, float ymax,
                                                    float zmin, float zmax,
                                                    float3 u, float3 v, float3 w); // Triangle
// Ray/Sphere intersection
__host__ __device__ float hit_ray_sphere(float3 ray_p, float3 ray_d,           // Ray
                                         float3 sphere_c, float sphere_rad);  // Sphere


// Ray/AABB intersection - Smits algorithm
__host__ __device__ float hit_ray_AABB(float3 ray_p, float3 ray_d,
                                       float aabb_xmin, float aabb_xmax,
                                       float aabb_ymin, float aabb_ymax,
                                       float aabb_zmin, float aabb_zmax);

// Ray/AABB test - Smits algorithm
__host__ __device__ bool test_ray_AABB(float3 ray_p, float3 ray_d,
                                       float aabb_xmin, float aabb_xmax,
                                       float aabb_ymin, float aabb_ymax,
                                       float aabb_zmin, float aabb_zmax);

// AABB/AABB test
__host__ __device__ bool test_AABB_AABB(float a_xmin, float a_xmax, float a_ymin, float a_ymax,
                                        float a_zmin, float a_zmax,
                                        float b_xmin, float b_xmax, float b_ymin, float b_ymax,
                                        float b_zmin, float b_zmax);

// Point/AABB test
__host__ __device__ bool test_point_AABB(float3 p,
                                         float aabb_xmin, float aabb_xmax,
                                         float aabb_ymin, float aabb_ymax,
                                         float aabb_zmin, float aabb_zmax);

// Ray/triangle intersection - Moller-Trumbore algorithm
__host__ __device__ float hit_ray_triangle(float3 ray_p, float3 ray_d,
                                           float3 tri_u,              // Triangle
                                           float3 tri_v,
                                           float3 tri_w);

#endif
