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

#ifndef BASE_OBJECT_CU
#define BASE_OBJECT_CU

#include "base_object.cuh"

BaseObject::BaseObject () {

    xmin = 0.0f;
    xmax = 0.0f;
    ymin = 0.0f;
    ymax = 0.0f;
    zmin = 0.0f;
    zmax = 0.0f;
    material_name = "MatName";
    object_name = "ObjNBame";

    // white by default
    color.r = 1.0;
    color.g = 1.0;
    color.b = 1.0;

    // Transparency by default
    transparency = 0.0;
}

void BaseObject::set_color(float r, float g, float b) {
    color.r = r;
    color.g = g;
    color.b = b;
}

void BaseObject::set_transparency(float val) {
    transparency = val;
}

void BaseObject::set_material(std::string mat_name) {
    material_name = mat_name;
}

void BaseObject::set_name(std::string obj_name) {
    object_name = obj_name;
}

#endif
