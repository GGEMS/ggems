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

#ifndef SPHERE_CU
#define SPHERE_CU

#include "sphere.h"

Sphere::Sphere(float ox, float oy, float oz, float rad,
               std::string mat_name) {

    cx = ox;
    cy = oy;
    cz = oz;
    radius = rad;
    material_name = mat_name;
    type = "sphere";
}

#endif
