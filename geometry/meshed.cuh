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
#include <float.h>

#include "base_object.cuh"
#include "G4SystemOfUnits.hh"
#include "vector.cuh"
#include "raytracing.cuh"
#include "constants.cuh"


#define NO_OCTREE 0
#define REG_OCTREE 1
//#define ADP_OCTREE 2


// Triangular-based meshed phantom
class Meshed : public BaseObject {
    public:
        Meshed();
        void load_from_raw(std::string filename);
        //void save_ggems_mesh(std::string filename);
        //void load_from_ggems_mesh(std::string filename);
        void build_regular_octree(unsigned int nx, unsigned int ny, unsigned int nz);

        void scale(float3 s);
        void scale(float sx, float sy, float sz);
        void rotate(float3 r);
        void rotate(float phi, float theta, float psi);
        void translate(float3 t);
        void translate(float tx, float ty, float tz);

        // Mesh data
        float *vertices;
        unsigned int number_of_triangles;
        unsigned int number_of_vertices;

        // Octree
        unsigned int nb_cell_x, nb_cell_y, nb_cell_z;
        unsigned short int octree_type;
        std::vector<float> nb_objs_per_cell;
        std::vector<float> list_objs_per_cell;
        std::vector<float> addr_to_cell;

    private:
};

#endif
