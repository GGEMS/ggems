// GGEMS Copyright (C) 2015

/*!
 * \file primitives.cuh
 * \brief Primitives geometry
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Tuesday May 31, 2016
 *
 */

#ifndef PRIMITIVES_CUH
#define PRIMITIVES_CUH

#include "global.cuh"
#include "vector.cuh"

// Struct that handle a triangle
struct f32triangle {
    f32xyz u, v, x;
};

// Struct that handle Obb data
struct AabbData {
    f32 xmin, xmax, ymin, ymax, zmin, zmax;  // AABB size
};

// Struct that handle Obb data
struct ObbData {
    f32 xmin, xmax, ymin, ymax, zmin, zmax;  // AABB size
    f32matrix44 transformation;              // Transformation matrix
    //f32xyz center;                           // OBB center
    //f32xyz u, v, w;                          // Absolute frame (OBB orthogonal space u, v, w)
};

// Struct that handle a meshed (not in SoA)
struct MeshedData {
    f32triangle *triangles;
    AabbData aabb;
    ui32 nb_of_triangles;
};

/* TODO: to remove
// Struct that handle CPU&GPU data
struct ObbVolume {
    ObbData data_h;   // Host
    ObbData data_d;   // Device
};
*/

#endif
