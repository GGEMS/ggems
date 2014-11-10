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
#include "raytracing.cuh"
#include "global.cuh"

#define AABB 0
#define SPHERE 1
#define MESHED 2
#define VOXELIZED 3

// Address of the header for the geometry structure
#define ADR_OBJ_TYPE 0
#define ADR_OBJ_MAT_ID 1
#define ADR_AABB_XMIN 2
#define ADR_AABB_XMAX 3
#define ADR_AABB_YMIN 4
#define ADR_AABB_YMAX 5
#define ADR_AABB_ZMIN 6
#define ADR_AABB_ZMAX 7

#define ADR_SPHERE_CX 8
#define ADR_SPHERE_CY 9
#define ADR_SPHERE_CZ 10
#define ADR_SPHERE_RADIUS 11

#define ADR_VOXELIZED_NX 8
#define ADR_VOXELIZED_NY 9
#define ADR_VOXELIZED_NZ 10
#define ADR_VOXELIZED_SX 11
#define ADR_VOXELIZED_SY 12
#define ADR_VOXELIZED_SZ 13
#define ADR_VOXELIZED_DATA 14

#define ADR_MESHED_NB_VERTICES 8
#define ADR_MESHED_NB_TRIANGLES 9
#define ADR_MESHED_OCTREE_TYPE 10
#define ADR_MESHED_OCTREE_NX 11
#define ADR_MESHED_OCTREE_NY 12
#define ADR_MESHED_OCTREE_NZ 13
#define ADR_MESHED_OCTREE_SX 14
#define ADR_MESHED_OCTREE_SY 15
#define ADR_MESHED_OCTREE_SZ 16
#define ADR_MESHED_DATA 17

#define SIZE_AABB_OBJ 8
#define SIZE_SPHERE_OBJ 12
#define SIZE_VOXELIZED_OBJ 14 // + number of voxels
#define SIZE_MESHED_OBJ 17 // + number of vertices * 3 (xyz) + octree

// Struct that handle the geometry of the world
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


// Host/Device function that handle geometry

unsigned int __host__ __device__ get_geometry_material(Scene geometry, unsigned int id_geom, float3 pos);
float __host__ __device__ get_distance_to_object(Scene geometry, unsigned int adr_geom, unsigned int obj_type,
                                                 float3 pos, float3 dir);
void __host__ __device__ get_next_geometry_boundary(Scene geometry, unsigned int cur_geom,
                                                     float3 pos, float3 dir,
                                                     float &interaction_distance,
                                                     unsigned int &geometry_volume);

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
        void print_tree();

        // Build the scene
        void build_scene();

        // Utils
        //void save_ggems_geometry(std::string filename);
        void print_geometry();
        //void print_raw_geometry();

        // World geometry description
        Scene world;
        std::vector<std::string> materials_list;   // List of the materials used
        std::vector<std::string> name_objects;     // Name of each object
        std::vector<Color> object_colors;          // Color of each object
        std::vector<float> object_transparency;    // Transparency of each object

    private:        
        unsigned int get_material_index(std::string material_name);
        void update_tree_address();

        // Store object temporally before initializing the complete geometry
        std::map<unsigned int, char> buffer_obj_type;
        std::map<unsigned int, Aabb> buffer_aabb;
        std::map<unsigned int, Sphere> buffer_sphere;
        std::map<unsigned int, Voxelized> buffer_voxelized;
        std::map<unsigned int, Meshed> buffer_meshed;

        // Build object into the scene structure
        void build_object(Aabb obj);
        void build_object(Sphere obj);
        void build_object(Voxelized obj);
        void build_object(Meshed obj);




};

#endif
