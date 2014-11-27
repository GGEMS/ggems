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

#include "global.cuh"

#include "base_object.cuh"
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
        void save_ggems_mesh(std::string filename);
        void load_from_ggems_mesh(std::string filename);
        void build_regular_octree(unsigned int nx, unsigned int ny, unsigned int nz);

        void scale(float3 s);
        void scale(f32 sx, f32 sy, f32 sz);
        void rotate(float3 r);
        void rotate(f32 phi, f32 theta, f32 psi);
        void translate(float3 t);
        void translate(f32 tx, f32 ty, f32 tz);

        // Mesh data
        f32 *vertices;
        unsigned int number_of_triangles;
        unsigned int number_of_vertices;

        // Octree
        unsigned int nb_cell_x, nb_cell_y, nb_cell_z;
        f32 cell_size_x, cell_size_y, cell_size_z;
        unsigned short int octree_type;
        //   Store only non-null value in order to compress the octree
        std::vector<f32> nb_objs_per_cell;
        std::vector<f32> list_objs_per_cell;
        std::vector<f32> addr_to_cell;

    private:
};

#endif
