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
#define ADR_OBJ_SENSITIVE 2
#define ADR_AABB_XMIN 3
#define ADR_AABB_XMAX 4
#define ADR_AABB_YMIN 5
#define ADR_AABB_YMAX 6
#define ADR_AABB_ZMIN 7
#define ADR_AABB_ZMAX 8

#define ADR_SPHERE_CX 9
#define ADR_SPHERE_CY 10
#define ADR_SPHERE_CZ 11
#define ADR_SPHERE_RADIUS 12

#define ADR_VOXELIZED_NX 9
#define ADR_VOXELIZED_NY 10
#define ADR_VOXELIZED_NZ 11
#define ADR_VOXELIZED_SX 12
#define ADR_VOXELIZED_SY 13
#define ADR_VOXELIZED_SZ 14
#define ADR_VOXELIZED_DATA 15

#define ADR_MESHED_NB_VERTICES 9
#define ADR_MESHED_NB_TRIANGLES 10
#define ADR_MESHED_OCTREE_TYPE 11
#define ADR_MESHED_OCTREE_NX 12
#define ADR_MESHED_OCTREE_NY 13
#define ADR_MESHED_OCTREE_NZ 14
#define ADR_MESHED_OCTREE_SX 15
#define ADR_MESHED_OCTREE_SY 16
#define ADR_MESHED_OCTREE_SZ 17
#define ADR_MESHED_DATA 18

#define SIZE_AABB_OBJ 9
#define SIZE_SPHERE_OBJ 13
#define SIZE_VOXELIZED_OBJ 15 // + number of voxels
#define SIZE_MESHED_OBJ 18 // + number of vertices * 3 (xyz) + octree

// Struct that handle the geometry of the world
struct Scene {

    // Object structure
    ui32* ptr_objects;     // Address to access to the different objects
    ui32* size_of_objects; // Size of each object
    f32* data_objects;           // Parameters of each primitive in the world

    // Tree structure
    ui32* ptr_nodes;       // Address to access the different nodes
    ui32* size_of_nodes;   // Size of each node (nb of others nodes connected)
    ui32* child_nodes;     // List of child nodes
    ui32* mother_node;     // List of mother nodes

    ui32 cur_node_id;      // current node id

    // Dimension of each vector
    ui32 ptr_objects_dim;
    ui32 size_of_objects_dim;
    ui32 data_objects_dim;
    ui32 ptr_nodes_dim;
    ui32 size_of_nodes_dim;
    ui32 child_nodes_dim;
    ui32 mother_node_dim;
};


// Host/Device function that handle geometry

__host__ __device__ ui32 get_geometry_material(Scene geometry, ui32 id_geom, f64xyz pos);
__host__ __device__ f64 get_distance_to_object(Scene geometry, ui32 adr_geom, ui32 obj_type,
                                               f64xyz pos, f64xyz dir);
__host__ __device__ void get_next_geometry_boundary(Scene geometry, ui32 cur_geom,
                                                    f64xyz pos, f64xyz dir,
                                                    f64 &interaction_distance,
                                                    ui32 &geometry_volume);

// This class is used to build the geometry
class GeometryBuilder {
    public:
        GeometryBuilder();

        // Geometry management
        ui32 add_world(Aabb obj);
        ui32 add_object(Aabb obj, ui32 mother_id);
        ui32 add_object(Sphere obj, ui32 mother_id);
        ui32 add_object(Meshed obj, ui32 mother_id);
        ui32 add_object(Voxelized obj, ui32 mother_id);

        // Hierarchical structure of the geometry
        void add_root();
        void add_node(ui32 mother_id);
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
        std::vector<f32> object_transparency;    // Transparency of each object
        std::vector<bool> object_wireframe;        // Wireframe option for each object

    private:        
        ui32 get_material_index(std::string material_name);
        void update_tree_address();

        // Store object temporally before initializing the complete geometry
        std::map<ui32, i8> buffer_obj_type;
        std::map<ui32, Aabb> buffer_aabb;
        std::map<ui32, Sphere> buffer_sphere;
        std::map<ui32, Voxelized> buffer_voxelized;
        std::map<ui32, Meshed> buffer_meshed;

        // Build object into the scene structure
        void build_object(Aabb obj);
        void build_object(Sphere obj);
        void build_object(Voxelized obj);
        void build_object(Meshed obj);




};

#endif
