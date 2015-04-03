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

#ifndef SPECT_HEAD_CU
#define SPECT_HEAD_CU

#include "spect_head.cuh"

SpectHead::SpectHead () {}

SpectHead::SpectHead (f32 ox, f32 oy, f32 oz,
            f32 sizex, f32 sizey, f32 sizez,
            std::string mat_name, std::string obj_name) {

    // Half size
    sizex *= 0.5;
    sizey *= 0.5;
    sizez *= 0.5;

    xmin = -sizex;
    xmax = sizex;
    ymin = -sizey;
    ymax = sizey;
    zmin = -sizez;
    zmax = sizez;
    
    material_name = mat_name;
    object_name = obj_name;
    
    // head gravity center
    obb_center.x = ox;
    obb_center.y = oy;
    obb_center.z = oz;

    // OBB frame
    u.x=1.0; u.y=0.0; u.z=0.0;
    v.x=0.0; v.y=1.0; v.z=0.0;
    w.x=0.0; w.y=0.0; w.z=1.0;
    //nb_ring_heads = 1;

}

/*void SpectHead::set_ring_repeater(ui32 nb_head) {
    nb_ring_heads = nb_head;
}*/




#endif
