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

#ifndef COLLI_H
#define COLLI_H

#include "vector.cuh"
#include "global.cuh"
#include "obb.cuh"

// Hexagon center coordinates, format data is defined as SoA
struct CoordHex2 {
        f32 *y;
        f32 *z;
        i32 size;
};

// Hexagonal Hole Collimator
class Colli : public Obb {
    public:
        Colli();
        Colli(f32 ox, f32 oy, f32 oz,
            f32 sizex, f32 sizey, f32 sizez,
            std::string hole_mat_name, std::string septa_mat_name, std::string obj_name);
        
        void set_height(f32 height);
        void set_radius(f32 radius);
        
        void set_cubic_repeater(i32 numx, i32 numy, i32 numz,
                                f32 vecx, f32 vecy, f32 vecz);
        
        void set_linear_repeater(f32 vecx, f32 vecy, f32 vecz);
        
        void build_colli();
        
        // Materials for hole and septa
        std::string hole_material_name;
        std::string septa_material_name;
        
        // Colli center
        //f32xyz colli_center;
        
        // Septa paramters
        f32 septa_height, hole_radius;
        
        // Cubic array repetition parameters
        i32xyz cubarray_repnum;
        f32xyz cubarray_repvec;
        // Linear repetition parameters
        f32xyz linear_repvec;
        
        CoordHex2 centerOfHexagons;
        
    private:
};

#endif
