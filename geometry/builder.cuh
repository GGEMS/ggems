// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef BUILDER_H
#define BUILDER_H

#include <vector>
#include <string>
#include "aabb.cuh"
#include "sphere.cuh"
#include "meshed.cuh"
#include "voxelized.cuh"

#define AABB 0
#define SPHERE 1
#define MESHED 2
#define VOXELIZED 3

// Class to manage the hierarchical structure of the world
class BVH {
    public:
        BVH();
        void add_root();
        void add_node(unsigned int mother_id);
        unsigned int get_current_id();
        void print();

        std::vector<unsigned int> ptr_nodes;
        std::vector<unsigned int> size_of_nodes;
        std::vector<unsigned int> child_nodes;
        std::vector<unsigned int> mother_node;
        unsigned int cur_node_id;

    private:
        void update_address();

};

// Class that handle the geometry of the world
class Geometry {
    public:
        BVH tree;                                  // Tree structure of the world
        std::vector<unsigned int> ptr_objects;     // Address to access to the different objects
        std::vector<unsigned int> size_of_objects; // Size of each object
        std::vector<float> data_objects;           // Parameters of each primitive in the world
                                                   // Type Material_ID Params1 Params2 ...
        std::vector<std::string> materials_list;   // List of the materials used
        std::vector<std::string> name_objects;     // Name of each object
};

// This class is used to build the geometry
class GeometryBuilder {
    public:
        GeometryBuilder();
        unsigned int add_world(Aabb obj);
        unsigned int add_object(Aabb obj, unsigned int mother_id);
        unsigned int add_object(Sphere obj, unsigned int mother_id);
        unsigned int add_object(Meshed obj, unsigned int mother_id);
        unsigned int add_object(Voxelized obj, unsigned int mother_id);

        void save_world(std::string filename);

        void print();
        void print_raw();

        Geometry World;

    private:        
        unsigned int get_material_index(std::string material_name);



};

#endif
