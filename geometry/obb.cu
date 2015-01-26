// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef OBB_CU
#define OBB_CU

#include "obb.cuh"

Obb::Obb () {}

Obb::Obb (f32 ox, f32 oy, f32 oz,
          f32 sizex, f32 sizey, f32 sizez,
          std::string mat_name, std::string obj_name) {

    // Half size
    sizex *= 0.5;
    sizey *= 0.5;
    sizez *= 0.5;

    // Init
    xmin = ox-sizex;
    xmax = ox+sizex;
    ymin = oy-sizey;
    ymax = oy+sizey;
    zmin = oz-sizez;
    zmax = oz+sizez;
    material_name = mat_name;
    object_name = obj_name;

    // OBB center
    obb_center.x = ox;
    obb_center.y = oy;
    obb_center.z = oz;

    // OBB frame
    u.x=1.0; u.y=0.0; u.z=0.0;
    v.x=0.0; v.y=1.0; v.z=0.0;
    w.x=0.0; w.y=0.0; w.z=1.0;

}

// Translation
void Obb::translate(f32 tx, f32 ty, f32 tz) {

    // The bounding box have to be translate
    xmin += tx;
    xmax += tx;
    ymin += ty;
    ymax += ty;
    zmin += tz;
    zmax += tz;

    // Idem for the center-of-gravity
    obb_center = fxyz_add(obb_center, make_f32xyz(tx, ty, tz));

//    printf("OBB center %f %f %f\n", obb_center.x, obb_center.y, obb_center.z);

}

// Rotation
void Obb::rotate(f32 ax, f32 ay, f32 az) {
    // Store angle values for the VRML viewer
    angle.x = ax;
    angle.y = ay;
    angle.z = az;

    // First around x
    u = fxyz_rotate_x_axis(u, angle.x);
    v = fxyz_rotate_x_axis(v, angle.x);
    w = fxyz_rotate_x_axis(w, angle.x);
    // then around y
    u = fxyz_rotate_y_axis(u, angle.y);
    v = fxyz_rotate_y_axis(v, angle.y);
    w = fxyz_rotate_y_axis(w, angle.y);
    // finally around z
    u = fxyz_rotate_z_axis(u, angle.z);
    v = fxyz_rotate_z_axis(v, angle.z);
    w = fxyz_rotate_z_axis(w, angle.z);

    // Rotate the OBB center (new position)
    obb_center = fxyz_rotate_x_axis(obb_center, angle.x);
    obb_center = fxyz_rotate_y_axis(obb_center, angle.y);
    obb_center = fxyz_rotate_z_axis(obb_center, angle.z);

//    printf("u %f %f %f\n", u.x, u.y, u.z);
//    printf("v %f %f %f\n", v.x, v.y, v.z);
//    printf("w %f %f %f\n", w.x, w.y, w.z);
//    printf("OBB center rot %f %f %f\n", obb_center.x, obb_center.y, obb_center.z);

}













#endif
