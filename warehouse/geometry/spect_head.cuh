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

#ifndef SPECT_HEAD_H
#define SPECT_HEAD_H

#include "vector.cuh"
#include "global.cuh"
#include "obb.cuh"

// SPECT head
class SpectHead : public Obb {
  public:
        SpectHead();
        SpectHead(f32 ox, f32 oy, f32 oz,
             f32 sizex, f32 sizey, f32 sizez,
             std::string mat_name, std::string obj_name);

        //void set_ring_repeater(ui32);
        
        //ui32 nb_ring_heads;
        
         // Head center
        //f32xyz head_center;
        
        // Rotation angle
        //f32xyz angle;
        
        // Absolute frame (OBB orthogonal space u, v, w)
        //f32xyz u, v, w;
        
        //f32xyz size;
        
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
        //void rotate(f32 ax, f32 ay, f32 az);

    private:
  
};

#endif
