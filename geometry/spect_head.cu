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

    xmin = ox-sizex;
    xmax = ox+sizex;
    ymin = oy-sizey;
    ymax = oy+sizey;
    zmin = oz-sizez;
    zmax = oz+sizez;
    material_name = mat_name;
    object_name = obj_name;

}

#endif
