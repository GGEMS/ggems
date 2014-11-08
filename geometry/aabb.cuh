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

#ifndef AABB_H
#define AABB_H

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "global.cuh"
#include "base_object.cuh"

// Axis-Aligned Bounding Box
class Aabb : public BaseObject {
    public:
        Aabb();
        Aabb(float ox, float oy, float oz,
             float halflx, float halfly, float halflz,
             std::string mat_name, std::string obj_name);

        //void set_xlength(float);
        //void set_ylength(float);
        //void set_zlength(float);
        //void set_length(float x, float y, float z);
        //void set_position(float x, float y, float z);
        //void set_length(float3);

        //void scale(float3 s);
        //void scale(float sx, float sy, float sz);
        //void translate(float3 t);
        //void translate(float tx, float ty, float tz);

    private:
};

#endif
