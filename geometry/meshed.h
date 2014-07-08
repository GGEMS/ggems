// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
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

#ifndef MESHED_H
#define MESHED_H

#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include "../maths/vector.h"
#include "../processes/constants.h"

//#ifndef TRIANGLE
//#define TRIANGLE
//struct Triangle {
//    float3 u, v, w;
//};
//#endif

// Raycasting
//__host__ __device__ float distance_to_triangle(float px, float py, float pz,
//                                               float dx, float dy, float dz);

// Triangular-based meshed phantom
class MeshedPhantom {
    public:
        MeshedPhantom();
        void load(std::string filename);
        void set_material(std::string matname);

        void scale(float3 s);
        void scale(float sx, float sy, float sz);
        void rotate(float3 r);
        void rotate(float phi, float theta, float psi);
        void translate(float3 t);
        void translate(float tx, float ty, float tz);

        std::vector<float> vertices;
        std::string material_name;

    private:
};

#endif
