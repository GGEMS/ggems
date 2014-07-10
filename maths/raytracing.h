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

#include <cfloat>
#include "vector.h"

#define EPSILON 1e-6f

// Overlap test return (short int):
//       -1 No interection         +1 Intersection

// Hit collision return (float):
//       -1 No collision            t >= 0 Distance of collision


// Overlapping test AABB/Triangle - Akenine-Moller algorithm
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

// Ray/triangle intersection - Moller-Trumbore algorithm
__host__ __device__ float hit_ray_triangle(float3 ray_p, float3 ray_d,
                                           float3 tri_u,              // Triangle
                                           float3 tri_v,
                                           float3 tri_w);

#endif
