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

#ifndef GEOMETRY_BUILDER_CUH
#define GEOMETRY_BUILDER_CUH

#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>
#include "aabb.cuh"
#include "sphere.cuh"
#include "meshed.cuh"
#include "voxelized.cuh"

#define AABB 0
#define SPHERE 1
#define MESHED 2
#define VOXELIZED 3

#define ADR_OBJ_TYPE 0
#define ADR_OBJ_MAT_ID 1

// Class that handle the geometry of the world
struct Scene {

    // Object structure
    unsigned int* ptr_objects;     // Address to access to the different objects
    unsigned int* size_of_objects; // Size of each object
    float* data_objects;           // Parameters of each primitive in the world

    // Tree structure
    unsigned int* ptr_nodes;       // Address to access the different nodes
    unsigned int* size_of_nodes;   // Size of each node (nb of others nodes connected)
    unsigned int* child_nodes;     // List of child nodes
    unsigned int* mother_node;     // List of mother nodes

    unsigned int cur_node_id;      // current node id

    // Dimension of each vector
    unsigned int ptr_objects_dim;
    unsigned int size_of_objects_dim;
    unsigned int data_objects_dim;
    unsigned int ptr_nodes_dim;
    unsigned int size_of_nodes_dim;
    unsigned int child_nodes_dim;
    unsigned int mother_node_dim;
};

// This class is used to build the geometry
class GeometryBuilder {
    public:
        GeometryBuilder();

        // Geometry management
        unsigned int add_world(Aabb obj);
        unsigned int add_object(Aabb obj, unsigned int mother_id);
        unsigned int add_object(Sphere obj, unsigned int mother_id);
        unsigned int add_object(Meshed obj, unsigned int mother_id);
        unsigned int add_object(Voxelized obj, unsigned int mother_id);

        // Hierarchical structure of the geometry
        void add_root();
        void add_node(unsigned int mother_id);
        unsigned int get_current_id();
        void print_tree();

        // Utils
        void save_ggems_geometry(std::string filename);
        void print_geometry();
        void print_raw_geometry();

        // World geometry description
        Scene world;
        std::vector<std::string> materials_list;   // List of the materials used
        std::vector<std::string> name_objects;     // Name of each object

    private:        
        unsigned int get_material_index(std::string material_name);
        void update_tree_address();
        void push_back(unsigned int* vector, unsigned int* dim, unsigned int val);
        void insert(unsigned int* vector, unsigned int* dim, unsigned int pos, unsigned int val);



};

#endif
