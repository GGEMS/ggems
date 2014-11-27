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

#include "global.cuh"
#include "base_object.cuh"

// Axis-Aligned Bounding Box
class Aabb : public BaseObject {
    public:
        Aabb();
        Aabb(f32 ox, f32 oy, f32 oz,
             f32 halflx, f32 halfly, f32 halflz,
             std::string mat_name, std::string obj_name);

        //void set_xlength(f32);
        //void set_ylength(f32);
        //void set_zlength(f32);
        //void set_length(f32 x, f32 y, f32 z);
        //void set_position(f32 x, f32 y, f32 z);
        //void set_length(float3);

        //void scale(float3 s);
        //void scale(f32 sx, f32 sy, f32 sz);
        //void translate(float3 t);
        //void translate(f32 tx, f32 ty, f32 tz);

    private:
};

#endif
