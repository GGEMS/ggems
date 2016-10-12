// GGEMS Copyright (C) 2015

/*!
 * \file raytracing.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef RAYTRACING_H
#define RAYTRACING_H

#include "vector.cuh"
#include "primitives.cuh"
#include "global.cuh"

/// Function with simple precision /////////////////////////////////////////////////////////

// Hit/overlap test return (bool):
//       0 No interection         1 Intersection

// Hit collision return (f32 or f64):
//        t >= 0 Distance of collision (if no collision t = FLT_MAX)


// AABB/Triangle test - Akenine-Moller algorithm
__host__ __device__ bool overlap_triangle_AABB(f32 xmin, f32 xmax,        // AABB
                                               f32 ymin, f32 ymax,
                                               f32 zmin, f32 zmax,
                                               f32xyz u, f32xyz v, f32xyz w); // Triangle

// Overlapping distance between ray and AABB
__host__ __device__ f32 dist_overlap_ray_AABB(f32xyz ray_p, f32xyz ray_d,
                                              f32 aabb_xmin, f32 aabb_xmax,
                                              f32 aabb_ymin, f32 aabb_ymax,
                                              f32 aabb_zmin, f32 aabb_zmax);

// Overlapping distance between ray and AABB
__host__ __device__ f32 dist_overlap_ray_AABB(f32xyz ray_p, f32xyz ray_d,
                                              f32 aabb_xmin, f32 aabb_xmax,
                                              f32 aabb_ymin, f32 aabb_ymax,
                                              f32 aabb_zmin, f32 aabb_zmax);

//// Ray/OBB intersection - Inspired by POVRAY
//__host__ __device__ f32 dist_overlap_ray_OBB(f32xyz ray_p, f32xyz ray_d,
//                                    f32 aabb_xmin, f32 aabb_xmax,
//                                    f32 aabb_ymin, f32 aabb_ymax,
//                                    f32 aabb_zmin, f32 aabb_zmax,
//                                    f32xyz obb_center,
//                                    f32xyz u, f32xyz v, f32xyz w); // OBB frame

// Ray/OBB intersection - Using transformation matrix
__host__ __device__ f32 dist_overlap_ray_OBB(f32xyz ray_p, f32xyz ray_d, const ObbData &obb );

// Ray/Sphere intersection
__host__ __device__ f32 hit_ray_sphere(f32xyz ray_p, f32xyz ray_d,        // Ray
                                       f32xyz sphere_c, f32 sphere_rad);  // Sphere

// Ray/AABB intersection - Smits algorithm
__host__ __device__ f32 hit_ray_AABB( f32xyz ray_p, f32xyz ray_d,
                                      f32 aabb_xmin, f32 aabb_xmax,
                                      f32 aabb_ymin, f32 aabb_ymax,
                                      f32 aabb_zmin, f32 aabb_zmax );
__host__ __device__ f32 hit_ray_AABB( f32xyz ray_p, f32xyz ray_d,
                                      const AabbData &aabb );

// Ray/AABB test - Smits algorithm
__host__ __device__ bool test_ray_AABB( f32xyz ray_p, f32xyz ray_d,
                                        f32 aabb_xmin, f32 aabb_xmax,
                                        f32 aabb_ymin, f32 aabb_ymax,
                                        f32 aabb_zmin, f32 aabb_zmax );
__host__ __device__ bool test_ray_AABB( f32xyz ray_p, f32xyz ray_d,
                                        const AabbData &aabb );

// AABB/AABB test
__host__ __device__ bool test_AABB_AABB(f32 a_xmin, f32 a_xmax, f32 a_ymin, f32 a_ymax,
                                        f32 a_zmin, f32 a_zmax,
                                        f32 b_xmin, f32 b_xmax, f32 b_ymin, f32 b_ymax,
                                        f32 b_zmin, f32 b_zmax);

// Point/AABB test
__host__ __device__ bool test_point_AABB(f32xyz p,
                                         f32 aabb_xmin, f32 aabb_xmax,
                                         f32 aabb_ymin, f32 aabb_ymax,
                                         f32 aabb_zmin, f32 aabb_zmax);

__host__ __device__ bool test_point_AABB( f32xyz p,
                                          const AabbData &aabb );

// Point/AABB test with tolerance
__host__ __device__ bool test_point_AABB_with_tolerance(f32xyz p,
                                                        f32 aabb_xmin, f32 aabb_xmax,
                                                        f32 aabb_ymin, f32 aabb_ymax,
                                                        f32 aabb_zmin, f32 aabb_zmax,
                                                        f32 tol);

__host__ __device__ bool test_point_AABB_with_tolerance(f32xyz p,
                                                        const AabbData &aabb,
                                                        f32 tol);

// Ray/triangle intersection - Moller-Trumbore algorithm
__host__ __device__ f32 hit_ray_triangle(f32xyz ray_p, f32xyz ray_d,
                                         f32xyz tri_u,              // Triangle
                                         f32xyz tri_v,
                                         f32xyz tri_w);

//// Ray/OBB intersection - Inspired by POVRAY
//__host__ __device__ f32 hit_ray_OBB( f32xyz ray_p, f32xyz ray_d,
//                                     f32 aabb_xmin, f32 aabb_xmax,
//                                     f32 aabb_ymin, f32 aabb_ymax,
//                                     f32 aabb_zmin, f32 aabb_zmax,
//                                     f32xyz obb_center,
//                                     f32xyz u, f32xyz v, f32xyz w ); // OBB frame

// Ray/OBB intersection - Using transformation matrix
__host__ __device__ f32 hit_ray_OBB( f32xyz ray_p, f32xyz ray_d, const ObbData &obb );

////////////////////////////////////////////////////////////////////////////////////////

/*

/// Function with double precision /////////////////////////////////////////////////////////

#ifndef SINGLE_PRECISION
    // Add function with double precision

// Hit/overlap test return (bool):
//       0 No interection         1 Intersection

// Hit collision return (f32 or f64):
//        t >= 0 Distance of collision (if no collision t = FLT_MAX)


// AABB/Triangle test - Akenine-Moller algorithm
__host__ __device__ bool overlap_triangle_AABB(f64 xmin, f64 xmax,        // AABB
                                               f64 ymin, f64 ymax,
                                               f64 zmin, f64 zmax,
                                               f64xyz u, f64xyz v, f64xyz w); // Triangle
// Overlapping distance between ray and AABB
__host__ __device__ f64 dist_overlap_ray_AABB(f64xyz ray_p, f64xyz ray_d,
                                              f64 aabb_xmin, f64 aabb_xmax,
                                              f64 aabb_ymin, f64 aabb_ymax,
                                              f64 aabb_zmin, f64 aabb_zmax);
// Ray/Sphere intersection
__host__ __device__ f64 hit_ray_sphere(f64xyz ray_p, f64xyz ray_d,        // Ray
                                       f64xyz sphere_c, f64 sphere_rad);  // Sphere

// Ray/AABB intersection - Smits algorithm
__host__ __device__ f64 hit_ray_AABB(f64xyz ray_p, f64xyz ray_d,
                                     f64 aabb_xmin, f64 aabb_xmax,
                                     f64 aabb_ymin, f64 aabb_ymax,
                                     f64 aabb_zmin, f64 aabb_zmax);

// Ray/AABB test - Smits algorithm
__host__ __device__ bool test_ray_AABB(f64xyz ray_p, f64xyz ray_d,
                                       f64 aabb_xmin, f64 aabb_xmax,
                                       f64 aabb_ymin, f64 aabb_ymax,
                                       f64 aabb_zmin, f64 aabb_zmax);

// AABB/AABB test
__host__ __device__ bool test_AABB_AABB(f64 a_xmin, f64 a_xmax, f64 a_ymin, f64 a_ymax,
                                        f64 a_zmin, f64 a_zmax,
                                        f64 b_xmin, f64 b_xmax, f64 b_ymin, f64 b_ymax,
                                        f64 b_zmin, f64 b_zmax);

// Point/AABB test
__host__ __device__ bool test_point_AABB(f64xyz p,
                                         f64 aabb_xmin, f64 aabb_xmax,
                                         f64 aabb_ymin, f64 aabb_ymax,
                                         f64 aabb_zmin, f64 aabb_zmax);

// Point/OBB test
__host__ __device__ bool test_point_OBB(f64xyz p,
                                        f64 aabb_xmin, f64 aabb_xmax,
                                        f64 aabb_ymin, f64 aabb_ymax,
                                        f64 aabb_zmin, f64 aabb_zmax,
                                        f64xyz obb_center,
                                        f64xyz u, f64xyz v, f64xyz w); // OBB frame

// Ray/triangle intersection - Moller-Trumbore algorithm
__host__ __device__ f64 hit_ray_triangle(f64xyz ray_p, f64xyz ray_d,
                                         f64xyz tri_u,              // Triangle
                                         f64xyz tri_v,
                                         f64xyz tri_w);

// Ray/Plane intersection (f64 version)
__host__ __device__ f64 hit_ray_plane(f64xyz ray_p, f64xyz ray_d,       
                                      f64xyz plane_p, f64xyz plane_n);

// Ray/OBB intersection - Inspired by POVRAY
__host__ __device__ f64 hit_ray_OBB(f64xyz ray_p, f64xyz ray_d,
                                    f64 aabb_xmin, f64 aabb_xmax,
                                    f64 aabb_ymin, f64 aabb_ymax,
                                    f64 aabb_zmin, f64 aabb_zmax,
                                    f64xyz obb_center,
                                    f64xyz u, f64xyz v, f64xyz w); // OBB frame

// Ray/Septa intersection
__host__ __device__ f64 hit_ray_septa(f64xyz p, f64xyz dir, f64 half_size_x, f64 radius,
                                      f64xyz colli_center, f64xyz colli_u, f64xyz colli_v, f64xyz w); // Colli frame

#endif

*/




#endif
