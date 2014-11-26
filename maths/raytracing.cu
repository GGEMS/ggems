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

#ifndef RAYTRACING_CU
#define RAYTRACING_CU

#include "raytracing.cuh"


// Overlapping test AABB/Triangle - Akenine-Moller algorithm
__host__ __device__ short int overlap_AABB_triangle(float xmin, float xmax,        // AABB
                                                    float ymin, float ymax,
                                                    float zmin, float zmax,
                                                    float3 u,  // Triangle
                                                    float3 v,
                                                    float3 w) {

    // Compute box center
    float3 c;
    c.x = (xmin + xmax) * 0.5f;
    c.y = (ymin + ymax) * 0.5f;
    c.z = (zmin + zmax) * 0.5f;

    // Compute halfsize
    float3 hsize;
    hsize.x = (xmax - xmin) * 0.5f; // TODO improve that by xmax-cx
    hsize.y = (ymax - ymin) * 0.5f;
    hsize.z = (zmax - zmin) * 0.5f;

    // Translate triangle
    u = f3_sub(u, c);
    v = f3_sub(v, c);
    w = f3_sub(w, c);

    // Compute triangle edges
    float3 e0 = f3_sub(v, u);
    float3 e1 = f3_sub(w, v);
    float3 e2 = f3_sub(u, w);

    //// The first 9 tests ///////////////////////////////////////////////////
    float p0, p1, p2, min, max, rad;

    float3 fe;
    fe.x = fabs(e0.x);
    fe.y = fabs(e0.y);
    fe.z = fabs(e0.z);

    // AXISTEST_X01(e0z, e0y, fez, fey)
    p0 = e0.z*u.y - e0.y*u.z;
    p2 = e0.z*w.y - e0.y*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return -1;}

    // AXISTEST_Y02(e0z, e0x, fez, fex)
    p0 = -e0.z*u.x + e0.x*u.z;
    p2 = -e0.z*w.x + e0.x*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return -1;}

    // AXISTEST_Z12(e0y, e0x, fey, fex)
    p1 = e0.y*v.x - e0.x*v.y;
    p2 = e0.y*w.x - e0.x*w.y;
    if (p2 < p1) {min=p2; max=p1;} else {min=p1; max=p2;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return -1;}

    fe.x = fabs(e1.x);
    fe.y = fabs(e1.y);
    fe.z = fabs(e1.z);

    // AXISTEST_X01(e1z, e1y, fez, fey)
    p0 = e1.z*u.y - e1.y*u.z;
    p2 = e1.z*w.y - e1.y*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return -1;}

    // AXISTEST_Y02(e1z, e1x, fez, fex)
    p0 = -e1.z*u.x + e1.x*u.z;
    p2 = -e1.z*w.x + e1.x*w.z;
    if (p0 < p2) {min=p0; max=p2;} else {min=p2; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return -1;}

    // AXISTEST_Z0 (e1y, e1x, fey, fex)
    p0 = e1.y*u.x - e1.x*u.y;
    p1 = e1.y*v.x - e1.x*v.y;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return -1;}

    fe.x = fabs(e2.x);
    fe.y = fabs(e2.y);
    fe.z = fabs(e2.z);

    // AXISTEST_X2 (e2z, e2y, fez, fey)
    p0 = e2.z*u.y - e2.y*u.z;
    p1 = e2.z*v.y - e2.y*v.z;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.z*hsize.y + fe.y*hsize.z;
    if (min > rad || max < -rad) {return -1;}

    // AXISTEST_Y1 (e2z, e2x, fez, fex)
    p0 = -e2.z*u.x + e2.x*u.z;
    p1 = -e2.z*v.x + e2.x*v.z;
    if (p0 < p1) {min=p0; max=p1;} else {min=p1; max=p0;}
    rad = fe.z*hsize.x + fe.x*hsize.z;
    if (min > rad || max < -rad) {return -1;}

    // AXISTEST_Z12(e2y, e2x, fey, fex)
    p1 = e2.y*v.x - e2.x*v.y;
    p2 = e2.y*w.x - e2.x*w.y;
    if (p2 < p1) {min=p2; max=p1;} else {min=p1; max=p2;}
    rad = fe.y*hsize.x + fe.x*hsize.y;
    if (min > rad || max < -rad) {return -1;}

    //// The next 3 tests ///////////////////////////////////////////////////

    // test in X-direction
    min = max = u.x;
    if (v.x < min) min=v.x; if (v.x > max) max=v.x;
    if (w.x < min) min=w.x; if (w.x > max) max=w.x;
    if (min > hsize.x || max < -hsize.x) {return -1;}

    // test in Y-direction
    min = max = u.y;
    if (v.y < min) min=v.y; if (v.y > max) max=v.y;
    if (w.y < min) min=w.y; if (w.y > max) max=w.y;
    if (min > hsize.y || max < -hsize.y) {return -1;}

    // test in Z-direction
    min = max = u.z;
    if (v.z < min) min=v.z; if (v.z > max) max=v.z;
    if (w.z < min) min=w.z; if (w.z > max) max=w.z;
    if (min > hsize.z || max < -hsize.z) {return -1;}

    //// The last tests ///////////////////////////////////////////////////

    // Compute the plane
    float3 n = f3_cross(e0, e1);

    // AABB/Plane
    float3 vmin, vmax;

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

    if (f3_dot(n, vmin) > 0.0f) {return -1;}

    if (f3_dot(n, vmax) >= 0.0f) {return 1;}

    return -1;
}

// Ray/Sphere intersection
__host__ __device__ float hit_ray_sphere(float3 ray_p, float3 ray_d,           // Ray
                                                float3 sphere_c, float sphere_rad) { // Sphere

    // Sphere defintion (center, rad)
    float3 m = f3_sub(ray_p, sphere_c);
    float  b = f3_dot(m, ray_d);
    float  c = f3_dot(m, m) - sphere_rad*sphere_rad;

    if (c > 0.0f && b > 0.0f) {return FLT_MAX;}

    float discr = b*b - c;
    if (discr < 0.0f) {return FLT_MAX;}

    float t = -b - sqrt(discr);
    if (t < 0.0f) {t = 0.0f;}

    return t;
}

// Ray/AABB intersection - Smits algorithm
__host__ __device__ float hit_ray_AABB(float3 ray_p, float3 ray_d,
                                       float aabb_xmin, float aabb_xmax,
                                       float aabb_ymin, float aabb_ymax,
                                       float aabb_zmin, float aabb_zmax) {

    float idx, idy, idz;
    float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -FLT_MAX;
    tmax =  FLT_MAX;

    // on x
    if (fabs(ray_d.x) < EPSILON6) {
        if (ray_p.x < aabb_xmin || ray_p.x > aabb_xmax) {return FLT_MAX;}
    } else {
        idx = 1.0f / ray_d.x;
        tmin = (aabb_xmin - ray_p.x) * idx;
        tmax = (aabb_xmax - ray_p.x) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return FLT_MAX;}
    }
    // on y
    if (fabs(ray_d.y) < EPSILON6) {
        if (ray_p.y < aabb_ymin || ray_p.y > aabb_ymax) {return FLT_MAX;}
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
        if (tmin > tmax) {return FLT_MAX;}
    }
    // on z
    if (fabs(ray_d.z) < EPSILON6) {
        if (ray_p.z < aabb_zmin || ray_p.z > aabb_zmax) {return FLT_MAX;}
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
        if (tmin > tmax) {return FLT_MAX;}
    }

    // Return the smaller positive value
    if (tmin < 0 && tmax < 0) return FLT_MAX;
    if (tmin < 0) {
        return tmax;
    } else {
        return tmin;
    }

}

// Ray/AABB intersection test - Smits algorithm
__host__ __device__ bool test_ray_AABB(float3 ray_p, float3 ray_d,
                                       float aabb_xmin, float aabb_xmax,
                                       float aabb_ymin, float aabb_ymax,
                                       float aabb_zmin, float aabb_zmax) {

    float idx, idy, idz;
    float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

    tmin = -FLT_MAX;
    tmax =  FLT_MAX;

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

// AABB/AABB test
__host__ __device__ bool test_AABB_AABB(float a_xmin, float a_xmax, float a_ymin, float a_ymax,
                                        float a_zmin, float a_zmax,
                                        float b_xmin, float b_xmax, float b_ymin, float b_ymax,
                                        float b_zmin, float b_zmax) {

    if (a_xmax < b_xmin || a_xmin > b_xmax) return false;
    if (a_ymax < b_ymin || a_ymin > b_ymax) return false;
    if (a_zmax < b_zmin || a_zmin > b_zmax) return false;

    return true;
}

// Point/AABB test
__host__ __device__ bool test_point_AABB(float3 p,
                                         float aabb_xmin, float aabb_xmax,
                                         float aabb_ymin, float aabb_ymax,
                                         float aabb_zmin, float aabb_zmax) {

    if (p.x < aabb_xmin || p.x > aabb_xmax) return false;
    if (p.y < aabb_ymin || p.y > aabb_ymax) return false;
    if (p.z < aabb_zmin || p.z > aabb_zmax) return false;

    return true;
}


// Ray/triangle intersection - Moller-Trumbore algorithm
__host__ __device__ float hit_ray_triangle(float3 ray_p, float3 ray_d,
                                                  float3 tri_u,              // Triangle
                                                  float3 tri_v,
                                                  float3 tri_w) {

    float3 e1 = f3_sub(tri_v, tri_u); // Find vector for 2 edges sharing
    float3 e2 = f3_sub(tri_w, tri_u);

    float3 pp = f3_cross(ray_d, e2);
    float  a  = f3_dot(e1, pp);
    if (a > -1.0e-05f && a < 1.0e-05f) {return FLT_MAX;} // no hit

    float f = 1.0f / a;

    float3 s = f3_sub(ray_p, tri_u);
    float  u = f * f3_dot(s, pp);
    if (u < 0.0f || u > 1.0f) {return FLT_MAX;}

    float3 q = f3_cross(s, e1);
    float  v = f * f3_dot(ray_d, q);
    if (v < 0.0f || (u+v) > 1.0f) {return FLT_MAX;}

    // Ray hit the triangle
    return f * f3_dot(e2, q);

}
























































#endif

