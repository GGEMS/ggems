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

#ifndef GEOMETRY_BUILDER_CU
#define GEOMETRY_BUILDER_CU

#include "geometry_builder.cuh"

/////////////////////////////////////////////////////////////////////////////////////
///////// Host/Device functions /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

// Function that return the material of a volume
__host__ __device__ ui32 get_geometry_material(Scene geometry, ui32 id_geom, f64xyz pos) {
    ui32 adr_geom = geometry.ptr_objects[id_geom];
    ui32 obj_type = (ui32)geometry.data_objects[adr_geom+ADR_OBJ_TYPE];

    if (obj_type != VOXELIZED) {
        return (ui32)geometry.data_objects[adr_geom+ADR_OBJ_MAT_ID];
    } else if (obj_type == VOXELIZED) {
        // Change particle frame (into voxelized volume)
        pos.x -= (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN]; // -= xmin
        pos.y -= (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN]; // -= ymin
        pos.z -= (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; // -= zmin
        // Get the voxel index
        ui32xyz ind;
        ind.x = (ui32)(pos.x / (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SX]); // / sx
        ind.y = (ui32)(pos.y / (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SY]); // / sy
        ind.z = (ui32)(pos.z / (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SZ]); // / sz
//        printf("Vos ind %i %i %i aabb %f %f, %f %f, %f %f\n", ind.x, ind.y, ind.z,
//               geometry.data_objects[adr_geom+ADR_AABB_XMIN],
//               geometry.data_objects[adr_geom+ADR_AABB_XMAX],
//               geometry.data_objects[adr_geom+ADR_AABB_YMIN],
//               geometry.data_objects[adr_geom+ADR_AABB_YMAX],
//               geometry.data_objects[adr_geom+ADR_AABB_ZMIN],
//               geometry.data_objects[adr_geom+ADR_AABB_ZMAX]);
        // Return material
        ui32 abs_ind = ind.z * (geometry.data_objects[adr_geom+ADR_VOXELIZED_NY]*geometry.data_objects[adr_geom+ADR_VOXELIZED_NX])
                                 + ind.y*geometry.data_objects[adr_geom+ADR_VOXELIZED_NX] + ind.x;
        //printf("Mat: %i\n", (ui32)geometry.data_objects[adr_geom+ADR_VOXELIZED_DATA+abs_ind]);
        return (ui32)geometry.data_objects[adr_geom+ADR_VOXELIZED_DATA+abs_ind];
    } else {
        return 0;
    }
}

// Get distance from an object
__host__ __device__ f64 get_distance_to_object(Scene geometry, ui32 adr_geom,
                                               ui32 obj_type, f64xyz pos, f64xyz dir) {

    f64 distance = F64_MAX;

    // The main AABB bounding box volume

    f64 aabb_xmin = (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN];
    f64 aabb_xmax = (f64)geometry.data_objects[adr_geom+ADR_AABB_XMAX];
    f64 aabb_ymin = (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN];
    f64 aabb_ymax = (f64)geometry.data_objects[adr_geom+ADR_AABB_YMAX];
    f64 aabb_zmin = (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN];
    f64 aabb_zmax = (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMAX];

    // AABB volume
    if (obj_type == AABB) {

        distance = hit_ray_AABB(pos, dir, aabb_xmin, aabb_xmax,
                                aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax);

    // Sphere volume
    } else if (obj_type == SPHERE) {

        // Read first sphere parameters
        f64xyz c = make_f64xyz((f64)geometry.data_objects[adr_geom+ADR_SPHERE_CX],
                               (f64)geometry.data_objects[adr_geom+ADR_SPHERE_CY],
                               (f64)geometry.data_objects[adr_geom+ADR_SPHERE_CZ]);
        f64 r = (f64)geometry.data_objects[adr_geom+ADR_SPHERE_RADIUS];

        distance = hit_ray_sphere(pos, dir, c, r);

    } else if (obj_type == VOXELIZED) {

        // Change particle frame (into voxelized volume)
        f64xyz posinvox;
        posinvox.x = pos.x - (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN]; // -= xmin
        posinvox.y = pos.y - (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN]; // -= ymin
        posinvox.z = pos.z - (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; // -= zmin
        // Get spacing
        f64xyz s;
        s.x = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SX];
        s.y = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SY];
        s.z = (f64)geometry.data_objects[adr_geom+ADR_VOXELIZED_SZ];
        // Get the voxel index
        ui32xyz ind;
        ind.x = (ui32)(posinvox.x / s.x);
        ind.y = (ui32)(posinvox.y / s.y);
        ind.z = (ui32)(posinvox.z / s.z);

        //printf("Ind %i %i %i\n", ind.x, ind.y, ind.z);

        f64 xmin, ymin, xmax, ymax, zmin, zmax;
        xmin = ind.x*s.x + aabb_xmin; xmax = xmin+s.x;
        ymin = ind.y*s.y + aabb_ymin; ymax = ymin+s.y;
        zmin = ind.z*s.z + aabb_zmin; zmax = zmin+s.z;

//        xmin = (dir.x > 0 && posinvox.x > (ind.x+1)*s.x-EPSILON3) ? (ind.x+1)*s.x+volxmin : ind.x*s.x+volxmin;
//        ymin = (dir.y > 0 && posinvox.y > (ind.y+1)*s.y-EPSILON3) ? (ind.y+1)*s.y+volymin : ind.y*s.y+volymin;
//        zmin = (dir.z > 0 && posinvox.z > (ind.z+1)*s.z-EPSILON3) ? (ind.z+1)*s.z+volzmin : ind.z*s.z+volzmin;
//        xmax = (dir.x < 0 && posinvox.x < xmin + EPSILON3) ? xmin-s.x : xmin+s.x;
//        ymax = (dir.y < 0 && posinvox.y < ymin + EPSILON3) ? ymin-s.y : ymin+s.y;
//        zmax = (dir.z < 0 && posinvox.z < zmin + EPSILON3) ? zmin-s.z : zmin+s.z;

        // Get the distance
        distance = hit_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax);

//        if ((distance > -EPSILON6 && distance < EPSILON6) || distance > 100000) {
//        //if (d64 > 100000) {

//            f64 safety = hit_ray_AABB(pos, dir, aabb_xmin, aabb_xmax,
//                                      aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax);

//            printf("::::: Pos %f %f %f\n", pos.x, pos.y, pos.z);
//            printf("::::: Org %f %f %f\n", aabb_xmin, aabb_ymin, aabb_zmin);
//            printf("::::: RefPos %f %f %f\n", posinvox.x, posinvox.y, posinvox.z);
//            printf("::::: Scl %f %f %f\n", s.x, s.y, s.z);
//            printf("::::: Ind %i %i %i\n", ind.x, ind.y, ind.z);
//            printf("::::: Vox %f %f, %f %f, %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);
//            printf("::::: Dist %2.20f\n", distance);
//            printf("::::: Safety %2.20f\n", safety);
//            f64 a = -8.000009;
//            f64 b = 296.0;
//            f64 c = a+b;
//            printf("----- test %2.20f\n", c);
//        }

    } else if (obj_type == MESHED) {
/*
        ui32 octree_type = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_TYPE];

        // Read first the bounding box
        f32 xmin = geometry.data_objects[adr_geom+ADR_AABB_XMIN];
        f32 xmax = geometry.data_objects[adr_geom+ADR_AABB_XMAX];
        f32 ymin = geometry.data_objects[adr_geom+ADR_AABB_YMIN];
        f32 ymax = geometry.data_objects[adr_geom+ADR_AABB_YMAX];
        f32 zmin = geometry.data_objects[adr_geom+ADR_AABB_ZMIN];
        f32 zmax = geometry.data_objects[adr_geom+ADR_AABB_ZMAX];

        // First check the bounding box that contains the mesh
        if (!test_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax)) return FLT_MAX;

        // If no octree first check every triangle
        distance = FLT_MAX;
        f32 tri_distance;
        if (octree_type == NO_OCTREE) {
            ui32 nb_tri = geometry.data_objects[adr_geom+ADR_MESHED_NB_TRIANGLES];
            ui32 i=0;
            while (i < nb_tri) {
                // Fetch a triangle
                ui32 ptr_tri = adr_geom+ADR_MESHED_DATA+ i*9; // 3 vertices of f32xyz
                f32xyz u = make_f32xyz(geometry.data_objects[ptr_tri],
                                       geometry.data_objects[ptr_tri+1],
                                       geometry.data_objects[ptr_tri+2]);
                f32xyz v = make_f32xyz(geometry.data_objects[ptr_tri+3],
                                       geometry.data_objects[ptr_tri+4],
                                       geometry.data_objects[ptr_tri+5]);
                f32xyz w = make_f32xyz(geometry.data_objects[ptr_tri+6],
                                       geometry.data_objects[ptr_tri+7],
                                       geometry.data_objects[ptr_tri+8]);
                // Get distance to this triangle
                tri_distance = hit_ray_triangle(pos, dir, u, v, w);
                if (tri_distance < distance) distance = tri_distance;

                ++i;
            }
        // If regular octree
        } else if (octree_type == REG_OCTREE) {

            //// First get the octree index

            // Change particle frame (into voxelized volume)
            f32xyz localpos;
            localpos.x = pos.x - geometry.data_objects[adr_geom+ADR_AABB_XMIN]; // -= xmin
            localpos.y = pos.y - geometry.data_objects[adr_geom+ADR_AABB_YMIN]; // -= ymin
            localpos.z = pos.z - geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; // -= zmin
            // Get spacing
            f32xyz s;
            s.x = geometry.data_objects[adr_geom+ADR_VOXELIZED_SX];
            s.y = geometry.data_objects[adr_geom+ADR_VOXELIZED_SY];
            s.z = geometry.data_objects[adr_geom+ADR_VOXELIZED_SZ];
            // Get the voxel index
            int3 ind;
            ind.x = (ui32)(localpos.x / s.x);
            ind.y = (ui32)(localpos.y / s.y);
            ind.z = (ui32)(localpos.z / s.z);

            // DDA algorithm

            f32xyz finc;
            finc.x = dir.x*s.x;
            finc.y = dir.y*s.y;
            finc.z = dir.z*s.z;
            f32xyz fpos;
            fpos.x = f32(ind.x);
            fpos.y = f32(ind.y);
            fpos.z = f32(ind.z);

            ui32 nb_tri = geometry.data_objects[adr_geom+ADR_MESHED_NB_TRIANGLES];
            ui32 nx = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_NX];
            ui32 ny = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_NY];
            ui32 nz = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_NZ];
            ui32 adr_octree = adr_geom+ADR_MESHED_DATA+ 9*nb_tri; // 3 vertices of f32xyz

            ui32 index = ind.z*nx*ny + ind.y*nx + ind.x;

            // DDA until to find triangles on an octree cell
            while (geometry.data_objects[adr_octree+index] == 0) {
                ind.x = (ui32)fpos.x;
                ind.y = (ui32)fpos.y;
                ind.z = (ui32)fpos.z;

                // check boundary
                if (ind.x <0 && ind.x >= nx &&
                    ind.y <0 && ind.y >= ny &&
                    ind.z <0 && ind.z >= nz) {
                    break;
                }

                // new index
                index = ind.z*nx*ny + ind.y*nx + ind.x;
                // iterate DDA line
                fpos = f3_add(fpos, finc);
            }

            // if no triangle where found
            if (geometry.data_objects[adr_octree+index] == 0) {
                return FLT_MAX;
            // else check every triangle contain of the octree cell
            } else {
                ui32 tri_per_cell = geometry.data_objects[adr_octree+index];
                ui32 adr_to_cell = adr_octree + (nx*ny*nz) + index;
                ui32 ptr_list_tri = adr_octree + 2*(nx*ny*nz) + geometry.data_objects[adr_to_cell];
                ui32 i=0;
                while (i < tri_per_cell) {
                    ui32 ptr_tri = geometry.data_objects[ptr_list_tri + i*9];

                    f32xyz u = make_f32xyz(geometry.data_objects[ptr_tri],
                                           geometry.data_objects[ptr_tri+1],
                                           geometry.data_objects[ptr_tri+2]);
                    f32xyz v = make_f32xyz(geometry.data_objects[ptr_tri+3],
                                           geometry.data_objects[ptr_tri+4],
                                           geometry.data_objects[ptr_tri+5]);
                    f32xyz w = make_f32xyz(geometry.data_objects[ptr_tri+6],
                                           geometry.data_objects[ptr_tri+7],
                                           geometry.data_objects[ptr_tri+8]);

                    // Get distance to this triangle
                    tri_distance = hit_ray_triangle(pos, dir, u, v, w);
                    if (tri_distance < distance) distance = tri_distance;

                    ++i;
                } // while
            } // if triangle

        } // if regoctree
*/
    } // if meshed

    return distance;
}

// Find the next geometry along the path of the particle
__host__ __device__ void get_next_geometry_boundary(Scene geometry, ui32 cur_geom,
                                                    f64xyz pos, f64xyz dir,
                                                    f64 &interaction_distance,
                                                    ui32 &geometry_volume) {

    geometry_volume = cur_geom;
    f64 distance;

    ////// Mother

    // First check the mother volume (particle escaping the volume)
    ui32 adr_geom = geometry.ptr_objects[cur_geom];
    ui32 obj_type = (ui32)geometry.data_objects[adr_geom+ADR_OBJ_TYPE];

    // Special case of voxelized volume where there are voxel boundary
    if (obj_type == VOXELIZED) {           
        // Volume bounding box
        f64 safety = get_distance_to_object(geometry, adr_geom, AABB, pos, dir);
        // Voxel boundary
        distance = get_distance_to_object(geometry, adr_geom, VOXELIZED, pos, dir);

        // If the safety is equal to distance (numerically very close espilon6) to the voxel
        // boundary it means, that the particle is escaping the volume.
        //printf("         Safety %e vox distance %e pos %f %f %f\n", safety, distance, pos.x, pos.y, pos.z);
        if (fabs(distance-safety) < EPSILON3) {
            geometry_volume = geometry.mother_node[cur_geom];
        } else {
            // Distance < safety = Still inside the volume
            geometry_volume = cur_geom;
        }


    // Any other volumes
    } else {
        distance = get_distance_to_object(geometry, adr_geom, obj_type, pos, dir);
        geometry_volume = geometry.mother_node[cur_geom];
    }
    // First intersection distance given by the current volume
    interaction_distance = distance;// + EPSILON3; // overshoot

    ////// Children

    // Then check every child contains in this node
    ui32 adr_node = geometry.ptr_nodes[cur_geom];
    ui32 offset_node = 0;
    ui32 id_child_geom;

    while (offset_node < geometry.size_of_nodes[cur_geom]) {

        // Child id
        id_child_geom = geometry.child_nodes[adr_node + offset_node];

        // Determine the type of the volume
        ui32 adr_child_geom = geometry.ptr_objects[id_child_geom];
        obj_type = (ui32)geometry.data_objects[adr_child_geom+ADR_OBJ_TYPE];

        // Special case for voxelized volume (check the outter boundary)
        if (obj_type == VOXELIZED) {
            // Volume bounding box
            distance = get_distance_to_object(geometry, adr_child_geom, AABB, pos, dir);
        } else {
            // Any other volumes
            distance = get_distance_to_object(geometry, adr_child_geom, obj_type, pos, dir);
        }

        if (distance <= interaction_distance) {
            interaction_distance = distance;// + EPSILON3; // overshoot
            geometry_volume = id_child_geom;
        }

        //printf("Daughter %i dist %f id %i\n", obj_type, distance, id_child_geom);

        ++offset_node;
    }
}

/////////////////////////////////////////////////////////////////////////////////////
///////// Geometry Builder class ////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

GeometryBuilder::GeometryBuilder() {

    // Init the size of the structure Geometry
    world.ptr_objects_dim = 0;
    world.size_of_objects_dim = 0;
    world.data_objects_dim = 0;
    world.ptr_nodes_dim = 0;
    world.size_of_nodes_dim = 0;
    world.child_nodes_dim = 0;
    world.mother_node_dim = 0;

    // Init the first node id
    world.cur_node_id = 0;
}

///// Private ////////////////////////////////////////////////////

// Update the tree address
void GeometryBuilder::update_tree_address() {
    world.ptr_nodes[0] = 0;
    ui32 i=1;
    while (i < world.ptr_nodes_dim) {
        world.ptr_nodes[i] = world.ptr_nodes[i-1] + world.size_of_nodes[i-1];
        ++i;
    }
}

// Search and return the material index for a given material name
ui32 GeometryBuilder::get_material_index(std::string material_name) {

    // Check if this material is already used, if it is return the corresponding index
    ui32 index = 0;
    while (index < materials_list.size()) {
        if (materials_list[index] == material_name) return index;
        ++index;
    }

    // If it is not, add a new entry into the material table
    index = materials_list.size();
    materials_list.push_back(material_name);

    return index;
}

///// Hierarchical structure of the geometry ////////////////////////

// Add the root
void GeometryBuilder::add_root() {

    array_push_back(&world.ptr_nodes, world.ptr_nodes_dim, 0);
    array_push_back(&world.size_of_nodes, world.size_of_nodes_dim, 0);
    array_push_back(&world.mother_node, world.mother_node_dim, 0);
    world.cur_node_id = 0;

}

// Add a node
void GeometryBuilder::add_node(ui32 mother_id) {
    // New node ID
    world.cur_node_id++;

    // Insert this object into the tree
    array_insert(&world.child_nodes, world.child_nodes_dim,
                 world.ptr_nodes[mother_id]+world.size_of_nodes[mother_id], world.cur_node_id);

    // Update the tree
    world.size_of_nodes[mother_id]++;
    array_push_back(&world.size_of_nodes, world.size_of_nodes_dim, 0);
    array_push_back(&world.ptr_nodes, world.ptr_nodes_dim, world.cur_node_id);
    array_push_back(&world.mother_node, world.mother_node_dim, mother_id);

    // Update tree address
    update_tree_address();
}

// Print the tree structure of the geometry
void GeometryBuilder::print_tree() {
    // print each node
    ui32 i = 0;
    ui32 j = 0;
    while (i < world.size_of_nodes_dim) {
        printf("(mother: %i)--[node: %i]--(childs: ", world.mother_node[i], i);
        j=0; while (j < world.size_of_nodes[i]) {
            printf("%i,", world.child_nodes[world.ptr_nodes[i]+j]);
            ++j;
        }
        printf(")\n");
        ++i;
    }
    printf("\n");
}

///// Utils ////////////////////////////////////////////////////////////////////////////////

// Print the current world
void GeometryBuilder::print_geometry() {
    // Print out the tree structure
    print_tree();

    // Print out every object name
    ui32 i;
    printf("List of object:\n");
    i=0; while (i < name_objects.size()) {
        printf("%i - %s\n", i, name_objects[i].c_str());
        ++i;
    }
    printf("\n");

    // Print out every material name
    printf("List of material:\n");
    i=0; while (i < materials_list.size()) {
        printf("%i - %s\n", i, materials_list[i].c_str());
        ++i;
    }
    printf("\n");

    // Print out each object contains on the tree
    i=0; while (i < world.ptr_objects_dim) {
        // Get obj address
        ui32 address_obj = world.ptr_objects[i];

        // Object name
        printf("::: %s :::\n", name_objects[i].c_str());

        // Same header for everyone
        ui32 type = (ui32)(world.data_objects[address_obj+ADR_OBJ_TYPE]);
        ui32 mat = (ui32)(world.data_objects[address_obj+ADR_OBJ_MAT_ID]);
        f32 xmin = world.data_objects[address_obj+ADR_AABB_XMIN];
        f32 xmax = world.data_objects[address_obj+ADR_AABB_XMAX];
        f32 ymin = world.data_objects[address_obj+ADR_AABB_YMIN];
        f32 ymax = world.data_objects[address_obj+ADR_AABB_YMAX];
        f32 zmin = world.data_objects[address_obj+ADR_AABB_ZMIN];
        f32 zmax = world.data_objects[address_obj+ADR_AABB_ZMAX];

        // Print information0
        switch (type) {
        case AABB:
            printf("type: AABB\n"); break;
        case SPHERE:
            printf("type: SPHERE\n"); break;
        } // switch

        printf("material: %s\n", materials_list[mat].c_str());
        printf("xmin: %f xmax: %f ymin: %f ymax: %f zmin: %f zmax: %f\n\n",
                xmin, xmax, ymin, ymax, zmin, zmax);


        ++i;
    } // while
}



/*
// Print out the geometry raw data
void GeometryBuilder::print_raw() {

    // Print out every object name
    ui32 i;
    printf("List of object [%lu]: ", World.name_objects.size());
    i=0; while (i < World.name_objects.size()) {
        printf("%s ", World.name_objects[i].c_str());
        ++i;
    }
    printf("\n\n");

    // Print out every material name
    printf("List of material [%lu]: ", World.materials_list.size());
    i=0; while (i < World.materials_list.size()) {
        printf("%s ", World.materials_list[i].c_str());
        ++i;
    }
    printf("\n\n");

    // Print out size of objects
    printf("Size of objects [%lu]: ", World.size_of_objects.size());
    i=0; while (i < World.size_of_objects.size()) {
        printf("%i ", World.size_of_objects[i]);
        ++i;
    }
    printf("\n\n");

    // Print out object addresses
    printf("Object addresses [%lu]: ", World.ptr_objects.size());
    i=0; while (i < World.ptr_objects.size()) {
        printf("%i ", World.ptr_objects[i]);
        ++i;
    }
    printf("\n\n");

    // Print out object data
    printf("Object data [%lu]: ", World.data_objects.size());
    i=0; while (i < World.data_objects.size()) {
        printf("%f ", World.data_objects[i]);
        ++i;
    }
    printf("\n\n");

}
*/

/*
// Save the world in order to share an use it later
void GeometryBuilder::save_ggems_geometry(std::string filename) {

    // check extension
    if (filename.size() < 10) {
        printf("Error, to export a ggems geometry, the exension must be '.ggems_geom'!\n");
        return;
    }
    std::string ext = filename.substr(filename.size()-10);
    if (ext!="ggems_geom") {
        printf("Error, to export a ggems geometry, the exension must be '.ggems_geom'!\n");
        return;
    }

    FILE *pfile = fopen(filename.c_str(), "wb");
    ui32 i, nb, tmp;

    // .: Tree :.  -  First export the tree that structure the world

    // 1. ptr_nodes [N, data]
    nb = World.tree.ptr_nodes.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    fwrite(World.tree.ptr_nodes.data(), nb, sizeof(ui32), pfile);

    // 2. size_of_nodes [N, data]
    nb = World.tree.size_of_nodes.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    fwrite(World.tree.size_of_nodes.data(), nb, sizeof(ui32), pfile);

    // 3. child_nodes [N, data]
    nb = World.tree.child_nodes.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    fwrite(World.tree.child_nodes.data(), nb, sizeof(ui32), pfile);

    // 4. mother_node [N, data]
    nb = World.tree.mother_node.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    fwrite(World.tree.mother_node.data(), nb, sizeof(ui32), pfile);

    // 5. cur_node_id [val]
    fwrite(&World.tree.cur_node_id, 1, sizeof(ui32), pfile);

    // .: World :.  -  Then export the world

    // 6. name_objects [N, data]
    nb = World.name_objects.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    i=0; while (i < nb) {
        tmp = World.name_objects[i].size();
        fwrite(&tmp, 1, sizeof(ui32), pfile);
        fwrite(World.name_objects[i].c_str(), World.name_objects[i].size(), sizeof(i8), pfile);
        ++i;
    }

    // 7. materials_list [N, data]
    nb = World.materials_list.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    i=0; while (i < nb) {
        tmp = World.materials_list[i].size();
        fwrite(&tmp, 1, sizeof(ui32), pfile);
        fwrite(World.materials_list[i].c_str(), World.materials_list[i].size(), sizeof(i8), pfile);
        ++i;
    }

    // 8. ptr_objects [N, data]
    nb = World.ptr_objects.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    fwrite(World.ptr_objects.data(), nb, sizeof(ui32), pfile);

    // 9. size_of_objects [N, data]
    nb = World.size_of_objects.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    fwrite(World.size_of_objects.data(), nb, sizeof(ui32), pfile);

    // 10. data_objects [N, data] (the big one!!!)
    nb = World.data_objects.size();
    fwrite(&nb, 1, sizeof(ui32), pfile);
    fwrite(World.data_objects.data(), nb, sizeof(f32), pfile);


    fclose(pfile);
}
*/


////
////////////////////// Object management ///////////////////////////////////////////////////
////
//
// !!!! Convention of the head of any object written in the world structure !!!!
//
// Object Type
//  array_push_back(world.data_objects, world.data_objects_dim, (f32)AABB);
// Material index
//  array_push_back(world.data_objects, world.data_objects_dim, (f32)get_material_index(obj.material_name));
// AABB parameters
//  array_push_back(world.data_objects, world.data_objects_dim, obj.xmin);
//  array_push_back(world.data_objects, world.data_objects_dim, obj.xmax);
//  array_push_back(world.data_objects, world.data_objects_dim, obj.ymin);
//  array_push_back(world.data_objects, world.data_objects_dim, obj.ymax);
//  array_push_back(world.data_objects, world.data_objects_dim, obj.zmin);
//  array_push_back(world.data_objects, world.data_objects_dim, obj.zmax);

// Add the world
ui32 GeometryBuilder::add_world(Aabb obj) {

    // Add the root tree
    add_root();

    // Put this object into buffer
    buffer_aabb[world.cur_node_id] = obj;
    buffer_obj_type[world.cur_node_id] = AABB;

    return world.cur_node_id;

}

// Add an AABB object into the world
ui32 GeometryBuilder::add_object(Aabb obj, ui32 mother_id) {

    // Add this object to the tree
    add_node(mother_id);

    // Put this object into buffer
    buffer_aabb[world.cur_node_id] = obj;
    buffer_obj_type[world.cur_node_id] = AABB;

    return world.cur_node_id;
}

// Add a Sphere object into the world
ui32 GeometryBuilder::add_object(Sphere obj, ui32 mother_id) {

    // Add this object to the tree
    add_node(mother_id);

    // Put this object into buffer
    buffer_sphere[world.cur_node_id] = obj;
    buffer_obj_type[world.cur_node_id] = SPHERE;

    return world.cur_node_id;
}

// Add a Voxelized object into the world
ui32 GeometryBuilder::add_object(Voxelized obj, ui32 mother_id) {

    // Add this object to the tree
    add_node(mother_id);

    // Put this object into buffer
    buffer_voxelized[world.cur_node_id] = obj;
    buffer_obj_type[world.cur_node_id] = VOXELIZED;

    return world.cur_node_id;
}

// Add a Meshed object into the world
ui32 GeometryBuilder::add_object(Meshed obj, ui32 mother_id) {

    // Add thid object to the tree
    add_node(mother_id);

    // Put this object into buffer
    buffer_meshed[world.cur_node_id] = obj;
    buffer_obj_type[world.cur_node_id] = MESHED;

    return world.cur_node_id;
}

////////////////////////////////////////////////////////////////////////

// Build AABB object into the scene structure
void GeometryBuilder::build_object(Aabb obj) {

    // Store the address to access to this object
    array_push_back(&world.ptr_objects, world.ptr_objects_dim, world.data_objects_dim);

    // Store the information of this object

    // Object Type
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)AABB);
    // Material index
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)get_material_index(obj.material_name));
     // AABB parameters
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmax);

    // Name of this object
    name_objects.push_back(obj.object_name);
    // Color of this object
    object_colors.push_back(obj.color);
    // Transparency of this object
    object_transparency.push_back(obj.transparency);
    // Wireframe option of this object
    object_wireframe.push_back(obj.wireframe);
    // Store the size of this object
    array_push_back(&world.size_of_objects, world.size_of_objects_dim, SIZE_AABB_OBJ);
}

// Build sphere object into the scene structure
void GeometryBuilder::build_object(Sphere obj) {
    // Store the address to access to this object
    array_push_back(&world.ptr_objects, world.ptr_objects_dim, world.data_objects_dim);

    // Store the information of this object

    // Object Type
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)SPHERE);
    // Material index
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)get_material_index(obj.material_name));
     // AABB parameters
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmax);
    // Sphere parameters
    array_push_back(&world.data_objects, world.data_objects_dim, obj.cx);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.cy);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.cz);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.radius);

    // Name of this object
    name_objects.push_back(obj.object_name);
    // Color of this object
    object_colors.push_back(obj.color);
    // Transparency of this object
    object_transparency.push_back(obj.transparency);
    // Wireframe option of this object
    object_wireframe.push_back(obj.wireframe);
    // Store the size of this object
    array_push_back(&world.size_of_objects, world.size_of_objects_dim, SIZE_SPHERE_OBJ);
}

// Build voxelized object into the scene structure
void GeometryBuilder::build_object(Voxelized obj) {
    // TODO
    // If optimizer, every object contains within the voxelized volume must be identified
    // For instance when considering YVAN navigator (BVH must be stored on the world), each
    // voxel contain ID of the child volume

    ///// First step
    // We need to merge and update the material ID according the current list of materials
    // Build a LUT to convert the old IDs in new ones
    std::vector<ui32> new_id;
    ui32 i = 0;
    while (i < obj.list_of_materials.size()) {
        new_id.push_back(get_material_index(obj.list_of_materials[i]));
        ++i;
    }

    // Now convert every material ID contains on the voxelized volume
    i=0; while (i < obj.number_of_voxels) {
        obj.data[i] = new_id[obj.data[i]];
        ++i;
    }
    /////

    // Store the address to access to this object
    array_push_back(&world.ptr_objects, world.ptr_objects_dim, world.data_objects_dim);

    // Store the information of this object

    // Object Type
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)VOXELIZED);
    // Material index
    array_push_back(&world.data_objects, world.data_objects_dim, -1.0f); // // Heterogeneous material
    // AABB parameters
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmax);
    // Parameters for this object
    array_push_back(&world.data_objects, world.data_objects_dim, obj.nb_vox_x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.nb_vox_y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.nb_vox_z);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.spacing_x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.spacing_y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.spacing_z);
    // Finally append voxelized data into the world
    array_append_array(&world.data_objects, world.data_objects_dim, &(obj.data), obj.number_of_voxels);

    // Name of this object
    name_objects.push_back(obj.object_name);
    // Color of this object
    object_colors.push_back(obj.color);
    // Transparency of this object
    object_transparency.push_back(obj.transparency);
    // Wireframe option of this object
    object_wireframe.push_back(obj.wireframe);
    // Store the size of this object
    array_push_back(&world.size_of_objects, world.size_of_objects_dim, obj.number_of_voxels+SIZE_VOXELIZED_OBJ);

}

// Build meshed object into the scene structure
void GeometryBuilder::build_object(Meshed obj) {

    // Store the address to access to this object
    array_push_back(&world.ptr_objects, world.ptr_objects_dim, world.data_objects_dim);

    // Store the information of this object

    // Object Type
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)MESHED);
    // Material index
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)get_material_index(obj.material_name));
    // AABB parameters
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmax);
    // Parameters for this object
    array_push_back(&world.data_objects, world.data_objects_dim, obj.number_of_vertices);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.number_of_triangles);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.octree_type);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.nb_cell_x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.nb_cell_y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.nb_cell_z);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.cell_size_x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.cell_size_y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.cell_size_z);

    // Append triangles into the world
    array_append_array(&world.data_objects, world.data_objects_dim, &obj.vertices, 3*obj.number_of_vertices); // xyz

    // Finally append the octree if defined
    if (obj.octree_type == REG_OCTREE) {
        // Append the number of objects per cell
        f32 *tmp = &obj.nb_objs_per_cell[0]; // create a pointer to append into the world
        array_append_array(&world.data_objects, world.data_objects_dim, &tmp, obj.nb_objs_per_cell.size());

        // Append the addr of each cell
        tmp = &obj.addr_to_cell[0];
        array_append_array(&world.data_objects, world.data_objects_dim, &tmp, obj.addr_to_cell.size());

        // Append the list of objects per cell
        tmp = &obj.list_objs_per_cell[0];
        array_append_array(&world.data_objects, world.data_objects_dim, &tmp, obj.list_objs_per_cell.size());
    }

    //////////////

    // Name of this object
    name_objects.push_back(obj.object_name);
    // Color of this object
    object_colors.push_back(obj.color);
    // Transparency of this object
    object_transparency.push_back(obj.transparency);
    // Wireframe option of this object
    object_wireframe.push_back(obj.wireframe);
    // Store the size of this object
    if (obj.octree_type == REG_OCTREE) {
        array_push_back(&world.size_of_objects, world.size_of_objects_dim, 3*obj.number_of_vertices + obj.nb_objs_per_cell.size() +
                                                                           obj.addr_to_cell.size() + obj.list_objs_per_cell.size() + SIZE_MESHED_OBJ);

    } else { // NO_OCTREE
        array_push_back(&world.size_of_objects, world.size_of_objects_dim, 3*obj.number_of_vertices+SIZE_MESHED_OBJ);
    }

    // Clear data of the octree
    obj.nb_objs_per_cell.clear();
    obj.addr_to_cell.clear();
    obj.list_objs_per_cell.clear();

}

// Build the complete scene
void GeometryBuilder::build_scene() {

    // Scan every object a build it to the scene structure

    ui32 i = 0;
    while (i < world.ptr_nodes_dim) {

        // AABB
        if (buffer_obj_type[i] == AABB) {
            build_object(buffer_aabb[i]);
        // Sphere
        } else if (buffer_obj_type[i] == SPHERE) {
            build_object(buffer_sphere[i]);
        // Voxelized
        } else if (buffer_obj_type[i] == VOXELIZED) {
            build_object(buffer_voxelized[i]);
        // Meshed
        } else if (buffer_obj_type[i] == MESHED) {
            build_object(buffer_meshed[i]);
        }

        ++i;
    }

}

#endif
