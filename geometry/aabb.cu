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

#ifndef AABB_CU
#define AABB_CU

#include "aabb.cuh"

Aabb::Aabb () {}

Aabb::Aabb (float ox, float oy, float oz,
            float halflx, float halfly, float halflz,
            std::string mat_name, std::string obj_name) {

    xmin = ox-halflx;
    xmax = ox+halflx;
    ymin = oy-halfly;
    ymax = oy+halfly;
    zmin = oz-halflz;
    zmax = oz+halflz;
    material_name = mat_name;
    object_name = obj_name;

}

#endif
