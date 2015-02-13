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

#ifndef COLLI_CU
#define COLLI_CU

#include "colli.cuh"

Colli::Colli () {}

Colli::Colli (f32 ox, f32 oy, f32 oz,
          f32 sizex, f32 sizey, f32 sizez,
          std::string hole_mat_name, std::string septa_mat_name, std::string obj_name) {

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
    
    hole_material_name = hole_mat_name;
    septa_material_name = septa_mat_name;
    
    object_name = obj_name;
    
    // colli gravity center
    colli_center.x = ox;
    colli_center.y = oy;
    colli_center.z = oz;

}

void Colli::set_height(f32 height) {
    septa_height = height;
}

void Colli::set_radius(f32 radius) {
    hole_radius = radius;
}

void Colli::set_cubic_repeater(i32 numx, i32 numy, i32 numz,
                               f32 vecx, f32 vecy, f32 vecz) {
    // Cubic array repetition parameters
    cubarray_repnum.x = numx;
    cubarray_repnum.y = numy;
    cubarray_repnum.z = numz;
       
    cubarray_repvec.x = vecx;
    cubarray_repvec.y = vecy;
    cubarray_repvec.z = vecz;
}

void Colli::set_linear_repeater(f32 vecx, f32 vecy, f32 vecz) {
    // Linear repetition parameters
    linear_repvec.x = vecx;
    linear_repvec.y = vecy;
    linear_repvec.z = vecz;
}
   
    
void Colli::build_colli(){    
    printf("Entering build_colli .... \n");
    // Memory allocation of centerOfHexagons
    int number_hexagons = (cubarray_repnum.y * cubarray_repnum.z) + ((cubarray_repnum.y - 1) * (cubarray_repnum.z - 1));
       
    printf("number_hexagons %d \n", number_hexagons);
    
    centerOfHexagons.size = number_hexagons;
    centerOfHexagons.y = (f32*)malloc(centerOfHexagons.size * sizeof(f32));
    centerOfHexagons.z = (f32*)malloc(centerOfHexagons.size * sizeof(f32));
    
     // CubArrayRep
    for( int i = 0; i < cubarray_repnum.z; ++i )
    {
        for( int j = 0; j < cubarray_repnum.y; ++j )   
        {       
                int index = 2*i*cubarray_repnum.y - i;
                centerOfHexagons.y[ index + j ] =
                ( ( ( cubarray_repnum.y - 1.0 ) / 2.0 ) - j ) * cubarray_repvec.y;
                
            centerOfHexagons.z[ index + j ] =
                ( ( ( cubarray_repnum.z - 1.0 ) / 2.0 ) - i ) * cubarray_repvec.z;
            
        }
    }
    
    // LinearRep
    for( int i = 0; i < cubarray_repnum.z - 1; ++i )
    {
        for( int j = 0; j < cubarray_repnum.y - 1; ++j )   
        {       
                int index = (2*i*cubarray_repnum.y - i) + j;
                
                centerOfHexagons.y[ index + cubarray_repnum.y ] = 
                        centerOfHexagons.y[ index ] - linear_repvec.y;
                        
            centerOfHexagons.z[ index + cubarray_repnum.y ] = 
                centerOfHexagons.z[ index ] - linear_repvec.z;
                
        }
       
    }
    printf("hexagone_y 0 %f 85138 %f \n", centerOfHexagons.y[0], centerOfHexagons.y[85138]);
    printf("hexagone_z 0 %f 85138 %f \n", centerOfHexagons.z[0], centerOfHexagons.z[85138]);
    printf("Leaving build_colli .... \n");
}

#endif
