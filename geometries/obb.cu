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


#ifndef OBB_CU
#define OBB_CU

#include "obb.cuh"

Obb::Obb () {

    // Default values

    volume.data_h.xmin = -1; volume.data_h.xmax = -1;
    volume.data_h.ymin = -1; volume.data_h.ymax = -1;
    volume.data_h.zmin = -1; volume.data_h.zmax = -1;

    volume.data_h.angle = make_f32xyz(-1, -1, -1);
    volume.data_h.translate = make_f32xyz(0, 0, 0);
    volume.data_h.center = make_f32xyz(0, 0, 0);
    volume.data_h.u = make_f32xyz(1, 0, 0);
    volume.data_h.v = make_f32xyz(0, 1, 0);
    volume.data_h.w = make_f32xyz(0, 0, 1);


}

void Obb::set_size(f32 lx, f32 ly, f32 lz) {

    f32 hlx = 0.5*lx;
    f32 hly = 0.5*ly;
    f32 hlz = 0.5*lz;

    volume.data_h.xmin = -hlx; volume.data_h.xmax = hlx;
    volume.data_h.ymin = -hly; volume.data_h.ymax = hly;
    volume.data_h.zmin = -hlz; volume.data_h.zmax = hlz;

}

void Obb::set_center_position(f32 x, f32 y, f32 z) {
    volume.data_h.center.x = x;
    volume.data_h.center.y = y;
    volume.data_h.center.z = z;
}

// Translation
void Obb::translate(f32 tx, f32 ty, f32 tz) {

    // The translation is not apply on the bounding box, because is defined locally
    // We just need to translate the center-of-gravity
    t = make_f32xyz(tx, ty, tz);
    volume.data_h.translate = t;
    volume.data_h.center = fxyz_add(volume.data_h.center, t);

}

// Rotation
void Obb::rotate(f32 ax, f32 ay, f32 az) {   

    volume.data_h.angle.x = ax;
    volume.data_h.angle.y = ay;
    volume.data_h.angle.z = az;

    f32xyz ut, vt, wt; // temp vars

    // First around x
    ut = fxyz_rotate_x_axis(ut, ax);
    vt = fxyz_rotate_x_axis(vt, ax);
    wt = fxyz_rotate_x_axis(wt, ax);
    // then around y
    ut = fxyz_rotate_y_axis(ut, ay);
    vt = fxyz_rotate_y_axis(vt, ay);
    wt = fxyz_rotate_y_axis(wt, ay);
    // finally around z
    ut = fxyz_rotate_z_axis(ut, az);
    vt = fxyz_rotate_z_axis(vt, az);
    wt = fxyz_rotate_z_axis(wt, az);

    // Store the new OBB frame
    volume.data_h.u = ut;
    volume.data_h.v = vt;
    volume.data_h.w = wt;

    // Rotate the OBB center (new position)
    volume.data_h.center = fxyz_rotate_x_axis(volume.data_h.center, ax);
    volume.data_h.center = fxyz_rotate_y_axis(volume.data_h.center, ay);
    volume.data_h.center = fxyz_rotate_z_axis(volume.data_h.center, az);

}

#endif
