// GGEMS Copyright (C) 2015

/*!
 * \file raytracing.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef RAYTRACING_CU
#define RAYTRACING_CU

#include "raytracing.cuh"

/// Function with simple precision /////////////////////////////////////////////////////////

// Overlapping test AABB/Triangle - Akenine-Moller algorithm (f32 version)
__host__ __device__ bool overlap_AABB_triangle(f32 xmin, f32 xmax,        // AABB
                                               f32 ymin, f32 ymax,
                                               f32 zmin, f32 zmax,
                                               f32xyz u,  // Triangle
                                               f32xyz v,
                                               f32xyz w) {

    // Compute box center
    f32xyz c;
    c.x = (xmin + xmax) * 0.5f;
    c.y = (ymin + ymax) * 0.5f;
    c.z = (zmin + zmax) * 0.5f;

    // Compute halfsize
    f32xyz hsize;
    hsize.x = (xmax - xmin) * 0.5f; // TODO improve that by xmax-cx
    hsize.y = (ymax - ymin) * 0.5f;
    hsize.z = (zmax - zmin) * 0.5f;

    // Translate triangle
    u = fxyz_sub(u, c);
    v = fxyz_sub(v, c);
    w = fxyz_sub(w, c);

    // Compute triangle edges
    f32xyz e0 = fxyz_sub(v, u);
    f32xyz e1 = fxyz_sub(w, v);
    f32xyz e2 = fxyz_sub(u, w);

    //// The first 9 tests ///////////////////////////////////////////////////
    f32 p0, p1, p2, min, max, rad;

    f32xyz fe;
    fe.x = fabs(e0.x);
    fe.y = fabs(e0.y);
    fe.z = fabs(e0.z);

    // AXISTEST_X01(e0z, e0y, fez, fey)
    p0 = e0.z*u.y - e0.y*u.z;
    p2 = e0.z*w.y - e0.y*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Y02(e0z, e0x, fez, fex)
    p0 = -e0.z*u.x + e0.x*u.z;
    p2 = -e0.z*w.x + e0.x*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Z12(e0y, e0x, fey, fex)
    p1 = e0.y*v.x - e0.x*v.y;
    p2 = e0.y*w.x - e0.x*w.y;
    if (p2 < p1) {min=p2; max=p1;} else {min=p1; max=p2;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return false;}

    fe.x = fabs(e1.x);
    fe.y = fabs(e1.y);
    fe.z = fabs(e1.z);

    // AXISTEST_X01(e1z, e1y, fez, fey)
    p0 = e1.z*u.y - e1.y*u.z;
    p2 = e1.z*w.y - e1.y*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Y02(e1z, e1x, fez, fex)
    p0 = -e1.z*u.x + e1.x*u.z;
    p2 = -e1.z*w.x + e1.x*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Z0 (e1y, e1x, fey, fex)
    p0 = e1.y*u.x - e1.x*u.y;
    p1 = e1.y*v.x - e1.x*v.y;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return false;}

    fe.x = fabs(e2.x);
    fe.y = fabs(e2.y);
    fe.z = fabs(e2.z);

    // AXISTEST_X2 (e2z, e2y, fez, fey)
    p0 = e2.z*u.y - e2.y*u.z;
    p1 = e2.z*v.y - e2.y*v.z;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Y1 (e2z, e2x, fez, fex)
    p0 = -e2.z*u.x + e2.x*u.z;
    p1 = -e2.z*v.x + e2.x*v.z;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Z12(e2y, e2x, fey, fex)
    p1 = e2.y*v.x - e2.x*v.y;
    p2 = e2.y*w.x - e2.x*w.y;
    if (p2 < p1) {min=p2; max=p1;} else {min=p1; max=p2;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return false;}

    //// The next 3 tests ///////////////////////////////////////////////////

    // test in X-direction
    min = max = u.x;
    if (v.x < min) min=v.x; if (v.x > max) max=v.x;
    if (w.x < min) min=w.x; if (w.x > max) max=w.x;
    if (min > hsize.x || max < -hsize.x) {return false;}

    // test in Y-direction
    min = max = u.y;
    if (v.y < min) min=v.y; if (v.y > max) max=v.y;
    if (w.y < min) min=w.y; if (w.y > max) max=w.y;
    if (min > hsize.y || max < -hsize.y) {return false;}

    // test in Z-direction
    min = max = u.z;
    if (v.z < min) min=v.z; if (v.z > max) max=v.z;
    if (w.z < min) min=w.z; if (w.z > max) max=w.z;
    if (min > hsize.z || max < -hsize.z) {return false;}

    //// The last tests ///////////////////////////////////////////////////

    // Compute the plane
    f32xyz n = fxyz_cross(e0, e1);

    // AABB/Plane
    f32xyz vmin, vmax;

    if (n.x > 0.0f) {
        vmin.x = -hsize.x - u.x;
        vmax.x =  hsize.x - u.x;
    } else {
        vmin.x =  hsize.x - u.x;
        vmax.x = -hsize.x - u.x;
    }
    if (n.y > 0.0f) {
        vmin.y = -hsize.y - u.y;
        vmax.y =  hsize.y - u.y;
    } else {
        vmin.y =  hsize.y - u.y;
        vmax.y = -hsize.y - u.y;
    }
    if (n.z > 0.0f) {
        vmin.z = -hsize.z - u.z;
        vmax.z =  hsize.z - u.z;
    } else {
        vmin.z =  hsize.z - u.z;
        vmax.z = -hsize.z - u.z;
    }

    if (fxyz_dot(n, vmin) > 0.0f) {return false;}

    if (fxyz_dot(n, vmax) >= 0.0f) {return true;}

    return false;
}

/*
// Ray/OBB intersection - Inspired by POVRAY (f32 version)
__host__ __device__ f32 dist_overlap_ray_OBB(f32xyz ray_p, f32xyz ray_d,
                                    f32 aabb_xmin, f32 aabb_xmax,
                                    f32 aabb_ymin, f32 aabb_ymax,
                                    f32 aabb_zmin, f32 aabb_zmax,
                                    f32xyz obb_center,
                                    f32xyz u, f32xyz v, f32xyz w) {

    // Transform the ray in OBB' space, then do AABB
    f32xyz ray_obb = fxyz_sub(ray_p, obb_center);
    ray_p.x = fxyz_dot(ray_obb, u);
    ray_p.y = fxyz_dot(ray_obb, v);
    ray_p.z = fxyz_dot(ray_obb, w);

    f32xyz dir;
    dir.x = fxyz_dot(ray_d, u);
    dir.y = fxyz_dot(ray_d, v);
    dir.z = fxyz_dot(ray_d, w);

    return dist_overlap_ray_AABB(ray_p, dir, aabb_xmin, aabb_xmax, aabb_ymin,
      aabb_ymax, aabb_zmin, aabb_zmax);
}
*/

__host__ __device__ f32 dist_overlap_ray_OBB( f32xyz ray_p, f32xyz ray_d, ObbData obb )
{
    // Get pos and dir in local OBB frame
    ray_p = fxyz_global_to_local_position( obb.transformation, ray_p );
    ray_d = fxyz_global_to_local_direction( obb.transformation, ray_d );

    // Then AABB test
    return dist_overlap_ray_AABB( ray_p, ray_d, obb.xmin, obb.xmax, obb.ymin, obb.ymax,
                                  obb.zmin, obb.zmax );
}


// Overlapping distance between ray and AABB - Smits algorithm (f32 version)
__host__ __device__ f32 dist_overlap_ray_AABB(f32xyz ray_p, f32xyz ray_d,
                                     f32 aabb_xmin, f32 aabb_xmax,
                                     f32 aabb_ymin, f32 aabb_ymax,
                                     f32 aabb_zmin, f32 aabb_zmax) {

    f32 idx, idy, idz;
    f32 tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -F32_MAX;
    tmax =  F32_MAX;

    // on x
    if (fabs(ray_d.x) < EPSILON6) {
        if (ray_p.x < aabb_xmin || ray_p.x > aabb_xmax) {return 0;}
    } else {
        idx = 1.0f / ray_d.x;
        tmin = (aabb_xmin - ray_p.x) * idx;
        tmax = (aabb_xmax - ray_p.x) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return 0;}
    }
    // on y
    if (fabs(ray_d.y) < EPSILON6) {
        if (ray_p.y < aabb_ymin || ray_p.y > aabb_ymax) {return 0;}
    } else {
        idy = 1.0f / ray_d.y;
        tymin = (aabb_ymin - ray_p.y) * idy;
        tymax = (aabb_ymax - ray_p.y) * idy;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
        if (tmin > tmax) {return 0;}
    }
    // on z
    if (fabs(ray_d.z) < EPSILON6) {
        if (ray_p.z < aabb_zmin || ray_p.z > aabb_zmax) {return 0;}
    } else {
        idz = 1.0f / ray_d.z;
        tzmin = (aabb_zmin - ray_p.z) * idz;
        tzmax = (aabb_zmax - ray_p.z) * idz;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
        if (tmin > tmax) {return 0;}
    }

    // Return overlap distance
    return tmax-tmin;

}


// Ray/Sphere intersection (f32 version)
__host__ __device__ f32 hit_ray_sphere(f32xyz ray_p, f32xyz ray_d,        // Ray
                                       f32xyz sphere_c, f32 sphere_rad) { // Sphere

    // Sphere defintion (center, rad)
    f32xyz m = fxyz_sub(ray_p, sphere_c);
    f32  b = fxyz_dot(m, ray_d);
    f32  c = fxyz_dot(m, m) - sphere_rad*sphere_rad;

    if (c > 0.0f && b > 0.0f) {return F32_MAX;}

    f32 discr = b*b - c;
    if (discr < 0.0f) {return F32_MAX;}

    f32 t = -b - sqrt(discr);
    if (t < 0.0f) {t = 0.0f;}

    return t;
}

// Ray/AABB intersection - Smits algorithm (f32 version)
__host__ __device__ f32 hit_ray_AABB(f32xyz ray_p, f32xyz ray_d,
                                     f32 aabb_xmin, f32 aabb_xmax,
                                     f32 aabb_ymin, f32 aabb_ymax,
                                     f32 aabb_zmin, f32 aabb_zmax) {

    f32 idx, idy, idz;
    f32 tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -F32_MAX;
    tmax =  F32_MAX;

    // on x
    if (fabs(ray_d.x) < EPSILON6) {
        if (ray_p.x < aabb_xmin || ray_p.x > aabb_xmax) {return F32_MAX;}
    } else {
        idx = 1.0f / ray_d.x;
        tmin = (aabb_xmin - ray_p.x) * idx;
        tmax = (aabb_xmax - ray_p.x) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return F32_MAX;}
    }
    // on y
    if (fabs(ray_d.y) < EPSILON6) {
        if (ray_p.y < aabb_ymin || ray_p.y > aabb_ymax) {return F32_MAX;}
    } else {
        idy = 1.0f / ray_d.y;
        tymin = (aabb_ymin - ray_p.y) * idy;
        tymax = (aabb_ymax - ray_p.y) * idy;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
        if (tmin > tmax) {return F32_MAX;}
    }
    // on z
    if (fabs(ray_d.z) < EPSILON6) {
        if (ray_p.z < aabb_zmin || ray_p.z > aabb_zmax) {return F32_MAX;}
    } else {
        idz = 1.0f / ray_d.z;
        tzmin = (aabb_zmin - ray_p.z) * idz;
        tzmax = (aabb_zmax - ray_p.z) * idz;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
        if (tmin > tmax) {return F32_MAX;}
    }

    // Return the smaller positive value diff to zero
    if (tmin < 0 && (tmax < 0 || tmax == 0)) return F32_MAX;
    if (tmin <= 0) {
        return tmax;
    } else {
        return tmin;
    }

}

__host__ __device__ f32 hit_ray_AABB( f32xyz ray_p, f32xyz ray_d,
                                      AabbData aabb )
{
    return hit_ray_AABB( ray_p, ray_d,
                         aabb.xmin, aabb.xmax, aabb.ymin, aabb.ymax,
                         aabb.zmin, aabb.zmax );
}

// Ray/AABB intersection test - Smits algorithm (f32 version)
__host__ __device__ bool test_ray_AABB( f32xyz ray_p, f32xyz ray_d,
                                        f32 aabb_xmin, f32 aabb_xmax,
                                        f32 aabb_ymin, f32 aabb_ymax,
                                        f32 aabb_zmin, f32 aabb_zmax ) {

    f32 idx, idy, idz;
    f32 tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -F32_MAX;
    tmax =  F32_MAX;

    // on x
    if (fabs(ray_d.x) < EPSILON6) {
        if (ray_p.x < aabb_xmin || ray_p.x > aabb_xmax) {return false;}
    } else {
        idx = 1.0f / ray_d.x;
        tmin = (aabb_xmin - ray_p.x) * idx;
        tmax = (aabb_xmax - ray_p.x) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return false;}
    }
    // on y
    if (fabs(ray_d.y) < EPSILON6) {
        if (ray_p.y < aabb_ymin || ray_p.y > aabb_ymax) {return false;}
    } else {
        idy = 1.0f / ray_d.y;
        tymin = (aabb_ymin - ray_p.y) * idy;
        tymax = (aabb_ymax - ray_p.y) * idy;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
        if (tmin > tmax) {return false;}
    }
    // on z
    if (fabs(ray_d.z) < EPSILON6) {
        if (ray_p.z < aabb_zmin || ray_p.z > aabb_zmax) {return false;}
    } else {
        idz = 1.0f / ray_d.z;
        tzmin = (aabb_zmin - ray_p.z) * idz;
        tzmax = (aabb_zmax - ray_p.z) * idz;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
        if (tmin > tmax) {return false;}
    }

    // Return the smaller positive value
    if (tmin < 0 && tmax < 0) return false;

    return true;
}

__host__ __device__ bool test_ray_AABB( f32xyz ray_p, f32xyz ray_d,
                                        AabbData aabb )
{
    return test_ray_AABB( ray_p, ray_d, aabb );
}

// AABB/AABB test (f32 version)
__host__ __device__ bool test_AABB_AABB(f32 a_xmin, f32 a_xmax, f32 a_ymin, f32 a_ymax,
                                        f32 a_zmin, f32 a_zmax,
                                        f32 b_xmin, f32 b_xmax, f32 b_ymin, f32 b_ymax,
                                        f32 b_zmin, f32 b_zmax) {

    if (a_xmax < b_xmin || a_xmin > b_xmax) return false;
    if (a_ymax < b_ymin || a_ymin > b_ymax) return false;
    if (a_zmax < b_zmin || a_zmin > b_zmax) return false;

    return true;
}

// Point/AABB test (f32 version)
__host__ __device__ bool test_point_AABB(f32xyz p,
                                         f32 aabb_xmin, f32 aabb_xmax,
                                         f32 aabb_ymin, f32 aabb_ymax,
                                         f32 aabb_zmin, f32 aabb_zmax) {

    if (p.x < aabb_xmin || p.x > aabb_xmax) return false;
    if (p.y < aabb_ymin || p.y > aabb_ymax) return false;
    if (p.z < aabb_zmin || p.z > aabb_zmax) return false;

    return true;
}

__host__ __device__ bool test_point_AABB( f32xyz p,
                                          AabbData aabb )

{
    return test_point_AABB( p, aabb.xmin, aabb.xmax, aabb.ymin, aabb.ymax, aabb.zmin, aabb.zmax );
}

// Point/AABB test with tolerance (f32 version)
__host__ __device__ bool test_point_AABB_with_tolerance(f32xyz p,
                                                        f32 aabb_xmin, f32 aabb_xmax,
                                                        f32 aabb_ymin, f32 aabb_ymax,
                                                        f32 aabb_zmin, f32 aabb_zmax,
                                                        f32 tol)
{
    if (p.x < aabb_xmin+tol || p.x > aabb_xmax-tol) return false;
    if (p.y < aabb_ymin+tol || p.y > aabb_ymax-tol) return false;
    if (p.z < aabb_zmin+tol || p.z > aabb_zmax-tol) return false;

    return true;
}

// Ray/triangle intersection - Moller-Trumbore algorithm (f32 version)
__host__ __device__ f32 hit_ray_triangle(f32xyz ray_p, f32xyz ray_d,
                                         f32xyz tri_u,              // Triangle
                                         f32xyz tri_v,
                                         f32xyz tri_w) {

    f32xyz e1 = fxyz_sub(tri_v, tri_u); // Find vector for 2 edges sharing
    f32xyz e2 = fxyz_sub(tri_w, tri_u);

    f32xyz pp = fxyz_cross(ray_d, e2);
    f32  a  = fxyz_dot(e1, pp);
    if (a > -1.0e-05f && a < 1.0e-05f) {return F32_MAX;} // no hit

    f32 f = 1.0f / a;

    f32xyz s = fxyz_sub(ray_p, tri_u);
    f32  u = f * fxyz_dot(s, pp);
    if (u < 0.0f || u > 1.0f) {return F32_MAX;}

    f32xyz q = fxyz_cross(s, e1);
    f32  v = f * fxyz_dot(ray_d, q);
    if (v < 0.0f || (u+v) > 1.0f) {return F32_MAX;}

    // Ray hit the triangle
    return f * fxyz_dot(e2, q);

}

// Ray/OBB intersection - Using transformation matrix
__host__ __device__ f32 hit_ray_OBB( f32xyz ray_p, f32xyz ray_d, ObbData obb )
{

    //ray_d.x = 1.0; ray_d.y = 0.0; ray_d.z = 0.0;

//    printf(" before pos: %f %f %f     dir %f %f %f\n", ray_p.x, ray_p.y, ray_p.z, ray_d.x, ray_d.y, ray_d.z);

//    printf("\n\n");
//    printf(" | %f %f %f %f\n", obb.transformation.m00, obb.transformation.m01, obb.transformation.m02, obb.transformation.m03 );
//    printf(" | %f %f %f %f\n", obb.transformation.m10, obb.transformation.m11, obb.transformation.m12, obb.transformation.m13 );
//    printf(" | %f %f %f %f\n", obb.transformation.m20, obb.transformation.m21, obb.transformation.m22, obb.transformation.m23 );
//    printf(" | %f %f %f %f\n", obb.transformation.m30, obb.transformation.m31, obb.transformation.m32, obb.transformation.m33 );

    // Get pos and dir in local OBB frame
    ray_p = fxyz_global_to_local_position( obb.transformation, ray_p );
    ray_d = fxyz_global_to_local_direction( obb.transformation, ray_d );

//    printf(" after pos: %f %f %f     dir %f %f %f\n", ray_p.x, ray_p.y, ray_p.z, ray_d.x, ray_d.y, ray_d.z);

    // Then AABB test
    return hit_ray_AABB( ray_p, ray_d, obb.xmin, obb.xmax, obb.ymin, obb.ymax,
                         obb.zmin, obb.zmax );

}



/*
// Ray/OBB intersection - Inspired by POVRAY (f32 version)
__host__ __device__ f32 hit_ray_OBB(f32xyz ray_p, f32xyz ray_d,
                                    f32 aabb_xmin, f32 aabb_xmax,
                                    f32 aabb_ymin, f32 aabb_ymax,
                                    f32 aabb_zmin, f32 aabb_zmax,
                                    f32xyz obb_center,
                                    f32xyz u, f32xyz v, f32xyz w) {

//    printf(" POS OBB %f %f %f\n", ray_p.x, ray_p.y, ray_p.z);

    // Transform the ray in OBB' space, then do AABB
    f32xyz ray_obb = fxyz_sub(ray_p, obb_center);

//    printf(" NEW POS OBB %f %f %f\n", ray_obb.x, ray_obb.y, ray_obb.z);

    ray_p.x = fxyz_dot(ray_obb, u);
    ray_p.y = fxyz_dot(ray_obb, v);
    ray_p.z = fxyz_dot(ray_obb, w);

    f32xyz dir;
    dir.x = fxyz_dot(ray_d, u);
    dir.y = fxyz_dot(ray_d, v);
    dir.z = fxyz_dot(ray_d, w);

    return hit_ray_AABB(ray_p, dir, aabb_xmin, aabb_xmax, aabb_ymin, aabb_ymax,
                        aabb_zmin, aabb_zmax);
}
*/

////////////////////////////////////////////////////////////////////////////////////////////

/*

/// Function with double precision /////////////////////////////////////////////////////////


#ifndef SINGLE_PRECISION
    // Add function with double precision

// Overlapping test AABB/Triangle - Akenine-Moller algorithm (f64 version)
__host__ __device__ bool overlap_AABB_triangle(f64 xmin, f64 xmax,        // AABB
                                               f64 ymin, f64 ymax,
                                               f64 zmin, f64 zmax,
                                               f64xyz u,  // Triangle
                                               f64xyz v,
                                               f64xyz w) {

    // Compute box center
    f64xyz c;
    c.x = (xmin + xmax) * 0.5f;
    c.y = (ymin + ymax) * 0.5f;
    c.z = (zmin + zmax) * 0.5f;

    // Compute halfsize
    f64xyz hsize;
    hsize.x = (xmax - xmin) * 0.5f; // TODO improve that by xmax-cx
    hsize.y = (ymax - ymin) * 0.5f;
    hsize.z = (zmax - zmin) * 0.5f;

    // Translate triangle
    u = fxyz_sub(u, c);
    v = fxyz_sub(v, c);
    w = fxyz_sub(w, c);

    // Compute triangle edges
    f64xyz e0 = fxyz_sub(v, u);
    f64xyz e1 = fxyz_sub(w, v);
    f64xyz e2 = fxyz_sub(u, w);

    //// The first 9 tests ///////////////////////////////////////////////////
    f64 p0, p1, p2, min, max, rad;

    f64xyz fe;
    fe.x = fabs(e0.x);
    fe.y = fabs(e0.y);
    fe.z = fabs(e0.z);

    // AXISTEST_X01(e0z, e0y, fez, fey)
    p0 = e0.z*u.y - e0.y*u.z;
    p2 = e0.z*w.y - e0.y*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Y02(e0z, e0x, fez, fex)
    p0 = -e0.z*u.x + e0.x*u.z;
    p2 = -e0.z*w.x + e0.x*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Z12(e0y, e0x, fey, fex)
    p1 = e0.y*v.x - e0.x*v.y;
    p2 = e0.y*w.x - e0.x*w.y;
    if (p2 < p1) {min=p2; max=p1;} else {min=p1; max=p2;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return false;}

    fe.x = fabs(e1.x);
    fe.y = fabs(e1.y);
    fe.z = fabs(e1.z);

    // AXISTEST_X01(e1z, e1y, fez, fey)
    p0 = e1.z*u.y - e1.y*u.z;
    p2 = e1.z*w.y - e1.y*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Y02(e1z, e1x, fez, fex)
    p0 = -e1.z*u.x + e1.x*u.z;
    p2 = -e1.z*w.x + e1.x*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Z0 (e1y, e1x, fey, fex)
    p0 = e1.y*u.x - e1.x*u.y;
    p1 = e1.y*v.x - e1.x*v.y;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return false;}

    fe.x = fabs(e2.x);
    fe.y = fabs(e2.y);
    fe.z = fabs(e2.z);

    // AXISTEST_X2 (e2z, e2y, fez, fey)
    p0 = e2.z*u.y - e2.y*u.z;
    p1 = e2.z*v.y - e2.y*v.z;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Y1 (e2z, e2x, fez, fex)
    p0 = -e2.z*u.x + e2.x*u.z;
    p1 = -e2.z*v.x + e2.x*v.z;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return false;}

    // AXISTEST_Z12(e2y, e2x, fey, fex)
    p1 = e2.y*v.x - e2.x*v.y;
    p2 = e2.y*w.x - e2.x*w.y;
    if (p2 < p1) {min=p2; max=p1;} else {min=p1; max=p2;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return false;}

    //// The next 3 tests ///////////////////////////////////////////////////

    // test in X-direction
    min = max = u.x;
    if (v.x < min) min=v.x; if (v.x > max) max=v.x;
    if (w.x < min) min=w.x; if (w.x > max) max=w.x;
    if (min > hsize.x || max < -hsize.x) {return false;}

    // test in Y-direction
    min = max = u.y;
    if (v.y < min) min=v.y; if (v.y > max) max=v.y;
    if (w.y < min) min=w.y; if (w.y > max) max=w.y;
    if (min > hsize.y || max < -hsize.y) {return false;}

    // test in Z-direction
    min = max = u.z;
    if (v.z < min) min=v.z; if (v.z > max) max=v.z;
    if (w.z < min) min=w.z; if (w.z > max) max=w.z;
    if (min > hsize.z || max < -hsize.z) {return false;}

    //// The last tests ///////////////////////////////////////////////////

    // Compute the plane
    f64xyz n = fxyz_cross(e0, e1);

    // AABB/Plane
    f64xyz vmin, vmax;

    if (n.x > 0.0f) {
        vmin.x = -hsize.x - u.x;
        vmax.x =  hsize.x - u.x;
    } else {
        vmin.x =  hsize.x - u.x;
        vmax.x = -hsize.x - u.x;
    }
    if (n.y > 0.0f) {
        vmin.y = -hsize.y - u.y;
        vmax.y =  hsize.y - u.y;
    } else {
        vmin.y =  hsize.y - u.y;
        vmax.y = -hsize.y - u.y;
    }
    if (n.z > 0.0f) {
        vmin.z = -hsize.z - u.z;
        vmax.z =  hsize.z - u.z;
    } else {
        vmin.z =  hsize.z - u.z;
        vmax.z = -hsize.z - u.z;
    }

    if (fxyz_dot(n, vmin) > 0.0f) {return false;}

    if (fxyz_dot(n, vmax) >= 0.0f) {return true;}

    return false;
}

// Overlapping distance between ray and AABB - Smits algorithm (f64 version)
__host__ __device__ f64 dist_overlap_ray_AABB(f64xyz ray_p, f64xyz ray_d,
                                     f64 aabb_xmin, f64 aabb_xmax,
                                     f64 aabb_ymin, f64 aabb_ymax,
                                     f64 aabb_zmin, f64 aabb_zmax) {

    f64 idx, idy, idz;
    f64 tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -F64_MAX;
    tmax =  F64_MAX;

    // on x
    if (fabs(ray_d.x) < EPSILON6) {
        if (ray_p.x < aabb_xmin || ray_p.x > aabb_xmax) {return 0;}
    } else {
        idx = 1.0f / ray_d.x;
        tmin = (aabb_xmin - ray_p.x) * idx;
        tmax = (aabb_xmax - ray_p.x) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return 0;}
    }
    // on y
    if (fabs(ray_d.y) < EPSILON6) {
        if (ray_p.y < aabb_ymin || ray_p.y > aabb_ymax) {return 0;}
    } else {
        idy = 1.0f / ray_d.y;
        tymin = (aabb_ymin - ray_p.y) * idy;
        tymax = (aabb_ymax - ray_p.y) * idy;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
        if (tmin > tmax) {return 0;}
    }
    // on z
    if (fabs(ray_d.z) < EPSILON6) {
        if (ray_p.z < aabb_zmin || ray_p.z > aabb_zmax) {return 0;}
    } else {
        idz = 1.0f / ray_d.z;
        tzmin = (aabb_zmin - ray_p.z) * idz;
        tzmax = (aabb_zmax - ray_p.z) * idz;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
        if (tmin > tmax) {return 0;}
    }

    // Return overlap distance
    return tmax-tmin;

}

// Ray/Sphere intersection (f64 version)
__host__ __device__ f64 hit_ray_sphere(f64xyz ray_p, f64xyz ray_d,        // Ray
                                       f64xyz sphere_c, f64 sphere_rad) { // Sphere

    // Sphere defintion (center, rad)
    f64xyz m = fxyz_sub(ray_p, sphere_c);
    f64  b = fxyz_dot(m, ray_d);
    f64  c = fxyz_dot(m, m) - sphere_rad*sphere_rad;

    if (c > 0.0f && b > 0.0f) {return F64_MAX;}

    f64 discr = b*b - c;
    if (discr < 0.0f) {return F64_MAX;}

    f64 t = -b - sqrt(discr);
    if (t < 0.0f) {t = 0.0f;}

    return t;
}

// Ray/Plane intersection (f64 version)
__host__ __device__ f64 hit_ray_plane(f64xyz ray_p, f64xyz ray_d,        // Ray
                                      f64xyz plane_p, f64xyz plane_n) { // Plane

    f64xyz m = fxyz_sub(plane_p, ray_p);
    f64 b = fxyz_dot(plane_n, m);
    f64 c = fxyz_dot(plane_n, ray_d);
    
    f64 t = b/c;
    
    if(t <= 0.0f) {return -F64_MAX;}

    return t;
}

// Ray/AABB intersection - Smits algorithm (f64 version)
__host__ __device__ f64 hit_ray_AABB(f64xyz ray_p, f64xyz ray_d,
                                     f64 aabb_xmin, f64 aabb_xmax,
                                     f64 aabb_ymin, f64 aabb_ymax,
                                     f64 aabb_zmin, f64 aabb_zmax) {

    f64 idx, idy, idz;
    f64 tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -F64_MAX;
    tmax =  F64_MAX;

    // on x
    if (fabs(ray_d.x) < EPSILON6) {
        if (ray_p.x < aabb_xmin || ray_p.x > aabb_xmax) {return F64_MAX;}
    } else {
        idx = 1.0f / ray_d.x;
        tmin = (aabb_xmin - ray_p.x) * idx;
        tmax = (aabb_xmax - ray_p.x) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return F64_MAX;}
    }
    // on y
    if (fabs(ray_d.y) < EPSILON6) {
        if (ray_p.y < aabb_ymin || ray_p.y > aabb_ymax) {return F64_MAX;}
    } else {
        idy = 1.0f / ray_d.y;
        tymin = (aabb_ymin - ray_p.y) * idy;
        tymax = (aabb_ymax - ray_p.y) * idy;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
        if (tmin > tmax) {return F64_MAX;}
    }
    // on z
    if (fabs(ray_d.z) < EPSILON6) {
        if (ray_p.z < aabb_zmin || ray_p.z > aabb_zmax) {return F64_MAX;}
    } else {
        idz = 1.0f / ray_d.z;
        tzmin = (aabb_zmin - ray_p.z) * idz;
        tzmax = (aabb_zmax - ray_p.z) * idz;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
        if (tmin > tmax) {return F64_MAX;}
    }

    // Return the smaller positive value diff to zero
    if (tmin < 0 && (tmax < 0 || tmax == 0)) return F64_MAX;
    if (tmin <= 0) {
        return tmax;
    } else {
        return tmin;
    }

}

// Ray/AABB intersection test - Smits algorithm (f64 version)
__host__ __device__ bool test_ray_AABB(f64xyz ray_p, f64xyz ray_d,
                                       f64 aabb_xmin, f64 aabb_xmax,
                                       f64 aabb_ymin, f64 aabb_ymax,
                                       f64 aabb_zmin, f64 aabb_zmax) {

    f64 idx, idy, idz;
    f64 tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -F64_MAX;
    tmax =  F64_MAX;

    // on x
    if (fabs(ray_d.x) < EPSILON6) {
        if (ray_p.x < aabb_xmin || ray_p.x > aabb_xmax) {return false;}
    } else {
        idx = 1.0f / ray_d.x;
        tmin = (aabb_xmin - ray_p.x) * idx;
        tmax = (aabb_xmax - ray_p.x) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return false;}
    }
    // on y
    if (fabs(ray_d.y) < EPSILON6) {
        if (ray_p.y < aabb_ymin || ray_p.y > aabb_ymax) {return false;}
    } else {
        idy = 1.0f / ray_d.y;
        tymin = (aabb_ymin - ray_p.y) * idy;
        tymax = (aabb_ymax - ray_p.y) * idy;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
        if (tmin > tmax) {return false;}
    }
    // on z
    if (fabs(ray_d.z) < EPSILON6) {
        if (ray_p.z < aabb_zmin || ray_p.z > aabb_zmax) {return false;}
    } else {
        idz = 1.0f / ray_d.z;
        tzmin = (aabb_zmin - ray_p.z) * idz;
        tzmax = (aabb_zmax - ray_p.z) * idz;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
        if (tmin > tmax) {return false;}
    }

    // Return the smaller positive value
    if (tmin < 0 && tmax < 0) return false;

    return true;
}

// AABB/AABB test (f64 version)
__host__ __device__ bool test_AABB_AABB(f64 a_xmin, f64 a_xmax, f64 a_ymin, f64 a_ymax,
                                        f64 a_zmin, f64 a_zmax,
                                        f64 b_xmin, f64 b_xmax, f64 b_ymin, f64 b_ymax,
                                        f64 b_zmin, f64 b_zmax) {

    if (a_xmax < b_xmin || a_xmin > b_xmax) return false;
    if (a_ymax < b_ymin || a_ymin > b_ymax) return false;
    if (a_zmax < b_zmin || a_zmin > b_zmax) return false;

    return true;
}

// Point/AABB test (f64 version)
__host__ __device__ bool test_point_AABB(f64xyz p,
                                         f64 aabb_xmin, f64 aabb_xmax,
                                         f64 aabb_ymin, f64 aabb_ymax,
                                         f64 aabb_zmin, f64 aabb_zmax) {

    if (p.x < aabb_xmin || p.x > aabb_xmax) return false;
    if (p.y < aabb_ymin || p.y > aabb_ymax) return false;
    if (p.z < aabb_zmin || p.z > aabb_zmax) return false;

    return true;
}

__host__ __device__ bool test_point_OBB(f64xyz p,
                                        f64 aabb_xmin, f64 aabb_xmax,
                                        f64 aabb_ymin, f64 aabb_ymax,
                                        f64 aabb_zmin, f64 aabb_zmax,
                                        f64xyz obb_center,
                                        f64xyz u, f64xyz v, f64xyz w) {
  
  //printf("BEFORE test_point_OBB: pos %f %f %f \n", p.x, p.y, p.z);
  //printf("OBB center pos %f %f %f \n", obb_center.x, obb_center.y, obb_center.z);
  
  // Transform the ray in OBB' space, then do AABB
  f64xyz ray_obb = fxyz_sub(p, obb_center);
  p.x = fxyz_dot(ray_obb, u);
  p.y = fxyz_dot(ray_obb, v);
  p.z = fxyz_dot(ray_obb, w);
  
  //printf("test_point_OBB: pos %f %f %f \n", p.x, p.y, p.z);
  
  return test_point_AABB(p, aabb_xmin, aabb_xmax, aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax);
}

// Ray/triangle intersection - Moller-Trumbore algorithm (f64 version)
__host__ __device__ f64 hit_ray_triangle(f64xyz ray_p, f64xyz ray_d,
                                         f64xyz tri_u,              // Triangle
                                         f64xyz tri_v,
                                         f64xyz tri_w) {

    f64xyz e1 = fxyz_sub(tri_v, tri_u); // Find vector for 2 edges sharing
    f64xyz e2 = fxyz_sub(tri_w, tri_u);

    f64xyz pp = fxyz_cross(ray_d, e2);
    f64  a  = fxyz_dot(e1, pp);
    if (a > -1.0e-06 && a < 1.0e-06) {return F64_MAX;} // no hit

    f64 f = 1.0 / a;

    f64xyz s = fxyz_sub(ray_p, tri_u);
    f64  u = f * fxyz_dot(s, pp);
    if (u < 0.0 || u > 1.0) {return F64_MAX;}

    f64xyz q = fxyz_cross(s, e1);
    f64  v = f * fxyz_dot(ray_d, q);
    if (v < 0.0 || (u+v) > 1.0) {return F64_MAX;}

    // Ray hit the triangle
    return f * fxyz_dot(e2, q);

}

// Ray/OBB intersection - Inspired by POVRAY (f64 version)
__host__ __device__ f64 hit_ray_OBB(f64xyz ray_p, f64xyz ray_d,
                                    f64 aabb_xmin, f64 aabb_xmax,
                                    f64 aabb_ymin, f64 aabb_ymax,
                                    f64 aabb_zmin, f64 aabb_zmax,
                                    f64xyz obb_center,
                                    f64xyz u, f64xyz v, f64xyz w) {

    // Transform the ray in OBB' space, then do AABB
    f64xyz ray_obb = fxyz_sub(ray_p, obb_center);
    ray_p.x = fxyz_dot(ray_obb, u);
    ray_p.y = fxyz_dot(ray_obb, v);
    ray_p.z = fxyz_dot(ray_obb, w);
    f64xyz dir;
    dir.x = fxyz_dot(ray_d, u);
    dir.y = fxyz_dot(ray_d, v);
    dir.z = fxyz_dot(ray_d, w);
    
   // printf("dir %f %f %f \n", dir.x, dir.y, dir.z);

    return hit_ray_AABB(ray_p, dir, aabb_xmin, aabb_xmax, aabb_ymin, aabb_ymax,
                        aabb_zmin, aabb_zmax);
}


// Ray/septa intersection
__host__ __device__ f64 hit_ray_septa(f64xyz p, f64xyz dir, f64 half_size_x, f64 radius,
                                         f64xyz colli_center, f64xyz colli_u, f64xyz colli_v, f64xyz colli_w) {
    
    f64 xmin, xmax, ymin, ymax, e1min, e1max, e2min, e2max;
    f64 tmin, tmax, tymin, tymax,  te1min, te1max, te2min, te2max, buf;
        
        
    
    //////// First, transform the ray in OBB' space
//    //f64xyz ray_obb = fxyz_sub(p, colli_center);
//    p.x = fxyz_dot(ray_obb, colli_u);
//    p.y = fxyz_dot(ray_obb, colli_v);
//    p.z = fxyz_dot(ray_obb, colli_w);
//    f64xyz dir;
//    dir.x = fxyz_dot(d, colli_u);
//    dir.y = fxyz_dot(d, colli_v);
//    dir.z = fxyz_dot(d, colli_w);
    //////////////////////////////////////
    
    //printf("hit septa: pos %f %f %f dir %f %f %f \n", p.x, p.y, p.z, dir.x, dir.y, dir.z);
    
    xmin = -half_size_x;
    xmax = half_size_x;
        
    ymin = e1min = e2min = -radius;
    ymax = e1max = e2max = radius;
        
    tmin = -F64_MAX;
    tmax = F64_MAX;
    
//     int w;
    
    f64xyz di;
        
        // on x
    if (fabs(dir.x) < EPSILON6) {
        if (p.x < xmin || p.x > xmax) {return 0.0;}
    }
    else {
//         w = 0;
        di.x = 1.0f / dir.x;
        tmin =  (xmin - p.x) * di.x;
        tmax = (xmax - p.x) * di.x;
       //printf("on x: %f %f - %f %f - %f %f \n", xmin, xmax, p.x, di.x, tmin, tmax);
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return 0.0;}
    }
    
    // on y
    if (fabs(dir.y) < EPSILON6) {
        if (p.y < ymin || p.y > ymax) {return 0.0;}
    }
    else {
        di.y = 1.0f / dir.y;
        tymin = (ymin - p.y) * di.y;
        tymax = (ymax - p.y) * di.y;
        //printf("on y: %f %f - %f %f - %f %f \n", ymin, ymax, p.y, di.y, tymin, tymax);
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax; }
        if (tmin > tmax) {return 0.0;}
    }
    
    // on e1  (changement de referentiel dans le plan yz, rotation de -60°) 
    
    f64 p1y = (p.y * cos( -M_PI / 3.0 )) + (p.z * sin ( -M_PI / 3.0 ));
    
    f64 d1y = dir.y * cos( -M_PI / 3.0 ) + dir.z * sin ( -M_PI / 3.0 );

   // printf("e1 p1y %f d1y %f \n", p1y, d1y);
    
    f64 di1y;
        
    if (fabs(d1y) < EPSILON6) {
        if (p1y < e1min || p1y > e1max) {return 0.0;}
    }
    else {
        di1y = 1.0f / d1y;
        te1min = (e1min - p1y) * di1y;
        te1max = (e1max - p1y) * di1y;
       // printf("on e1: %f %f - %f %f - %f %f \n", e1min, e1max, p1y, d1y, te1min, te1max);
        if (te1min > te1max) {
            buf = te1min;
            te1min = te1max;
            te1max = buf;
        }
        if (te1min > tmin) {tmin = te1min;}
        if (te1max < tmax) {tmax = te1max;}
        if (tmin > tmax) {return 0.0;}
    }

        // on e2 (changement de referentiel dans le plan yz, rotation de +60°) 
            
    f64 p2y = (p.y * cos( M_PI / 3.0 )) + (p.z * sin ( M_PI / 3.0 )); 
     
    f64 d2y = dir.y * cos( M_PI / 3.0 ) + dir.z * sin ( M_PI / 3.0 );
    
   // printf("e2 p2y %f d2y %f \n", p2y, d2y);

    f64 di2y;
        
    if (fabs(d2y) < EPSILON6) {
    if (p2y < e2min || p2y > e2max) {return 0.0;}
    }
    else {
        di2y = 1.0f / d2y;
        te2min = (e2min - p2y) * di2y;
        te2max = (e2max - p2y) * di2y;
       // printf("on e2: %f %f - %f %f - %f %f \n", e2min, e2max, p2y, d2y, te2min, te2max);
        if (te2min > te2max) {
            buf = te2min;
            te2min = te2max;
            te2max = buf;
        }
        if (te2min > tmin) {tmin = te2min;}
        if (te2max < tmax) {tmax = te2max; }
        if (tmin > tmax) {return 0.0;}
    }
    
    return tmax;
}



#endif

*/


#endif

