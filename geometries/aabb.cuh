// GGEMS Copyright (C) 2015

/*!
 * \file aabb.cuh
 * \brief AABB geometry
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday May 30, 2016
 *
 */


#ifndef AABB_CUH
#define AABB_CUH

#include "global.cuh"

// Struct that handle Obb data
struct AabbData {
    f32 xmin, xmax, ymin, ymax, zmin, zmax;  // AABB size
};

#endif
