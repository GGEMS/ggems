// GGEMS Copyright (C) 2015

/*!
 * \file obb.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 *
 *
 */


#ifndef OBB_CUH
#define OBB_CUH

#include "global.cuh"
#include "vector.cuh"

// Struct that handle Obb data
struct ObbData {
    f32 xmin, xmax, ymin, ymax, zmin, zmax;  // AABB size
    f32xyz angle;                            // Rotation angle around each axis
    f32xyz translate;                        // Translation vector
    f32xyz center;                           // OBB center
    f32xyz u, v, w;                          // Absolute frame (OBB orthogonal space u, v, w)
    f32xyz size;                             // Size
};

// Struct that handle CPU&GPU data
struct ObbVolume {
    ObbData data_h;   // Host
    ObbData data_d;   // Device
};


// Oriented Bounding Box
class Obb {
    public:
        Obb();        

        //void set_xlength(f32);
        //void set_ylength(f32);
        //void set_zlength(f32);
        void set_size(f32 lx, f32 ly, f32 lz);
        void set_center_position(f32 x, f32 y, f32 z);
        //void set_length(float3);

        //void scale(float3 s);
        //void scale(f32 sx, f32 sy, f32 sz);

        //void translate(float3 t);
        void translate(f32 tx, f32 ty, f32 tz);
        void rotate(f32 ax, f32 ay, f32 az);

        // OBB volume
        ObbVolume volume;

    private:



};

#endif
