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

// Function that return if an object is sensitive or not
__host__ __device__ bool get_geometry_is_sensitive(Scene geometry, ui32 cur_geom) {
    ui32 adr_geom = geometry.ptr_objects[cur_geom];
    return (bool)geometry.data_objects[adr_geom+ADR_OBJ_SENSITIVE];
}

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

    // OBB volume
    } else if (obj_type == OBB) {

        f64xyz obb_center;
        obb_center.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_X];
        obb_center.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Y];
        obb_center.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_CENTER_Z];
        f64xyz u, v, w;
        u.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UX];
        u.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UY];
        u.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_UZ];
        v.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VX];
        v.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VY];
        v.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_VZ];
        w.x = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WX];
        w.y = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WY];
        w.z = (f64)geometry.data_objects[adr_geom+ADR_OBB_FRAME_WZ];

        distance = hit_ray_OBB(pos, dir, aabb_xmin, aabb_xmax,
                               aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax,
                               obb_center, u, v, w);
        //printf("Pos %f %f %f dir %f %f %f C %f %f %f OBB distance: %e\n", pos.x, pos.y, pos.z,
        //       dir.x, dir.y, dir.z, obb_center.x, obb_center.y, obb_center.z, distance); // DEBUG

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

        ui32 octree_type = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_TYPE];

        // First check the bounding box that contains the mesh
        if (!test_ray_AABB(pos, dir, aabb_xmin, aabb_xmax,
                           aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax)) return F64_MAX;

        // If no octree first check every triangle
        distance = F64_MAX;
        f64 tri_distance;

        if (octree_type == NO_OCTREE) {
            ui32 nb_tri = geometry.data_objects[adr_geom+ADR_MESHED_NB_TRIANGLES];
            ui32 i=0;
            while (i < nb_tri) {
                // Fetch a triangle
                ui32 ptr_tri = adr_geom+ADR_MESHED_DATA+ i*9; // 3 vertices of f32xyz
                f64xyz u = make_f64xyz((f64)geometry.data_objects[ptr_tri],
                                       (f64)geometry.data_objects[ptr_tri+1],
                                       (f64)geometry.data_objects[ptr_tri+2]);
                f64xyz v = make_f64xyz((f64)geometry.data_objects[ptr_tri+3],
                                       (f64)geometry.data_objects[ptr_tri+4],
                                       (f64)geometry.data_objects[ptr_tri+5]);
                f64xyz w = make_f64xyz((f64)geometry.data_objects[ptr_tri+6],
                                       (f64)geometry.data_objects[ptr_tri+7],
                                       (f64)geometry.data_objects[ptr_tri+8]);
                // Get distance to this triangle
                tri_distance = hit_ray_triangle(pos, dir, u, v, w);
                // Select the min positive value
                if (tri_distance >= 0 && tri_distance < distance) distance = tri_distance;

                ++i;
            }
            //printf("Mesh dist %2.10f\n", distance);
        // If regular octree
        } else if (octree_type == REG_OCTREE) {

            //// Compute the two point use to perform the raycast within the octree

            // If inside the octree, use the current position as entry point
            // else get the entry point that intersect the bouding box
            f64xyz entry_pt, exit_pt;

            // Inside
            if (test_point_AABB(pos, aabb_xmin, aabb_xmax, aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax)) {
                entry_pt = pos;
            // Outside
            } else {
                f64 distance_to_in = hit_ray_AABB(pos, dir, aabb_xmin, aabb_xmax,
                                                  aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax);
                entry_pt = fxyz_add(pos, fxyz_scale(dir, distance_to_in));
            }

            // Get the exit point
            f64 distance_to_out = hit_ray_AABB(entry_pt, dir, aabb_xmin, aabb_xmax,
                                               aabb_ymin, aabb_ymax, aabb_zmin, aabb_zmax);
            // Exception when the ray hit one of the AABB edge or corner (entry = exit point)
            //if (distance_to_out == F64_MAX ) return distance; // FIXME
            if (distance_to_out == DBL_MAX) {printf("EDGE\n"); return distance;}

            exit_pt = fxyz_add(entry_pt, fxyz_scale(dir, distance_to_out));

            //// Convert point into octree index

            // Get spacing
            f64xyz s;
            s.x = (f64)geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_SX];
            s.y = (f64)geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_SY];
            s.z = (f64)geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_SZ];

            // Change the frame
            f64xyz entry_ind;
            entry_ind.x = entry_pt.x - (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN]; // -= xmin
            entry_ind.y = entry_pt.y - (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN]; // -= ymin
            entry_ind.z = entry_pt.z - (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; // -= zmin
            f64xyz exit_ind;
            exit_ind.x = exit_pt.x - (f64)geometry.data_objects[adr_geom+ADR_AABB_XMIN]; // -= xmin
            exit_ind.y = exit_pt.y - (f64)geometry.data_objects[adr_geom+ADR_AABB_YMIN]; // -= ymin
            exit_ind.z = exit_pt.z - (f64)geometry.data_objects[adr_geom+ADR_AABB_ZMIN]; // -= zmin

            // Get the octree index
            entry_ind.x /= s.x;
            entry_ind.y /= s.y;
            entry_ind.z /= s.z;
            exit_ind.x /= s.x;
            exit_ind.y /= s.y;
            exit_ind.z /= s.z;

            // Cast index while entry/exit point is on the last slice (must be < nx | ny | nz)
            ui32 nx = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_NX];
            ui32 ny = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_NY];
            ui32 nz = geometry.data_objects[adr_geom+ADR_MESHED_OCTREE_NZ];

            if (entry_ind.x >= nx) entry_ind.x = nx-1;
            if (entry_ind.y >= ny) entry_ind.y = ny-1;
            if (entry_ind.z >= nz) entry_ind.z = nz-1;

            if (exit_ind.x >= nx) exit_ind.x = nx-1;
            if (exit_ind.y >= ny) exit_ind.y = ny-1;
            if (exit_ind.z >= nz) exit_ind.z = nz-1;

            //// Cross the octree with a raycast (DDA algorithm)

            ui32 jump = ny*nx;
            ui32 bigjump = jump*nz;
            ui32 nb_tri = geometry.data_objects[adr_geom+ADR_MESHED_NB_TRIANGLES];
            ui32 adr_octree = adr_geom+ADR_MESHED_DATA+ 9*nb_tri; // 3 vertices of f32xyz

            f64xyz diff = fxyz_sub(exit_ind, entry_ind);
            f64xyz l = fxyz_abs(diff);
            ui32 length = (ui32)l.y;
            if (l.x > length) length=(ui32)l.x;
            if (l.z > length) length=(ui32)l.z;
            f64 flength = 1.0 / (f64)length;
            f64xyz finc = fxyz_scale(diff, flength);
            f64xyz curf = entry_ind;

            ui16xyz curi;
            ui32 index;

            // Loop over the ray that cross the octree
            ui16 i=0; while (i < length) {

                // Get current index
                curi.x=(ui16)curf.x; curi.y=(ui16)curf.y; curi.z=(ui16)curf.z;
                index = curi.z*jump+curi.y*nx+curi.x;

                // If any triangle is found inside the current octree cell
                if (geometry.data_objects[adr_octree+index] != 0) {

                    ui32 tri_per_cell = (ui32)geometry.data_objects[adr_octree+index];
                    // bigjump => skip NbObjsPerCell data
                    ui32 adr_to_cell = adr_octree + bigjump + index;
                    // 2*bigjump = > skip NbObjsPerCell and AddrToCell data
                    ui32 ptr_list_tri = adr_octree + 2*bigjump + (ui32)geometry.data_objects[adr_to_cell];

                    ui32 icell=0; while (icell < tri_per_cell) {
                        //                                       9 vertices x Triangle index
                        ui32 ptr_tri = adr_geom+ADR_MESHED_DATA+ 9*(ui32)geometry.data_objects[ptr_list_tri + icell];
                        f64xyz u = make_f64xyz((f64)geometry.data_objects[ptr_tri],
                                               (f64)geometry.data_objects[ptr_tri+1],
                                               (f64)geometry.data_objects[ptr_tri+2]);
                        f64xyz v = make_f64xyz((f64)geometry.data_objects[ptr_tri+3],
                                               (f64)geometry.data_objects[ptr_tri+4],
                                               (f64)geometry.data_objects[ptr_tri+5]);
                        f64xyz w = make_f64xyz((f64)geometry.data_objects[ptr_tri+6],
                                               (f64)geometry.data_objects[ptr_tri+7],
                                               (f64)geometry.data_objects[ptr_tri+8]);
                        // Get distance to this triangle
                        tri_distance = hit_ray_triangle(pos, dir, u, v, w);

                        // Select the min positive value
                        if (tri_distance >= 0 && tri_distance < distance) distance = tri_distance;

                        ++icell;
                    } // while triangle
                }

                // Iterate the ray
                curf = fxyz_add(curf, finc);

                ++i;
            } // while raycast

        } // if regoctree

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

/*
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
*/


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

// Add a Obb object into the world
ui32 GeometryBuilder::add_object(Obb obj, ui32 mother_id) {

    // Add thid object to the tree
    add_node(mother_id);

    // Put this object into buffer
    buffer_obb[world.cur_node_id] = obj;
    buffer_obj_type[world.cur_node_id] = OBB;

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
    // Object sensitive
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)obj.sensitive);
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
    // Object sensitive
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)obj.sensitive);
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
    f32 *newdata = (f32*)malloc(sizeof(f32) * obj.number_of_voxels);
    i=0; while (i < obj.number_of_voxels) {
        newdata[i] = new_id[obj.data[i]];
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
    // Object sensitive
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)obj.sensitive);
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
    array_append_array(&world.data_objects, world.data_objects_dim, &(newdata), obj.number_of_voxels);

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

    // Free memory
    free(newdata);
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
    // Object sensitive
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)obj.sensitive);
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
    //printf("====> @Vertices %i\n", world.data_objects_dim); // DEBUG
    //printf("====>  Vertice @200 tri: %f\n", obj.vertices[9*200]);
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
        //printf("====> @ListObjs %i cal@40 %f %f %f\n", world.data_objects_dim, obj.list_objs_per_cell[40],
        //       obj.list_objs_per_cell[41], obj.list_objs_per_cell[42]);

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

// Build OBB object into the scene structure
void GeometryBuilder::build_object(Obb obj) {

    // Store the address to access to this object
    array_push_back(&world.ptr_objects, world.ptr_objects_dim, world.data_objects_dim);

    // Store the information of this object

    // Object Type
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)OBB);
    // Material index
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)get_material_index(obj.material_name));
    // Object sensitive
    array_push_back(&world.data_objects, world.data_objects_dim, (f32)obj.sensitive);
    // AABB parameters
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.xmax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.ymax);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmin);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.zmax);
    // OBB center
    array_push_back(&world.data_objects, world.data_objects_dim, obj.obb_center.x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.obb_center.y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.obb_center.z);
    //printf("Build OBB center %f %f %f\n", obj.obb_center.x, obj.obb_center.y, obj.obb_center.z);
    // OBB frame
    array_push_back(&world.data_objects, world.data_objects_dim, obj.u.x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.u.y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.u.z);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.v.x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.v.y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.v.z);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.w.x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.w.y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.w.z);
    // Rotation angle on each axis (in deg)
    array_push_back(&world.data_objects, world.data_objects_dim, obj.angle.x);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.angle.y);
    array_push_back(&world.data_objects, world.data_objects_dim, obj.angle.z);

    // Name of this object
    name_objects.push_back(obj.object_name);
    // Color of this object
    object_colors.push_back(obj.color);
    // Transparency of this object
    object_transparency.push_back(obj.transparency);
    // Wireframe option of this object
    object_wireframe.push_back(obj.wireframe);
    // Store the size of this object
    array_push_back(&world.size_of_objects, world.size_of_objects_dim, SIZE_OBB_OBJ);
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
        // OBB
        } else if (buffer_obj_type[i] == OBB) {
            build_object(buffer_obb[i]);
        }

        ++i;
    }

}

// Copy the complete scene to the GPU
void GeometryBuilder::copy_scene_cpu2gpu() {

    // First allocate the GPU mem for the scene
    HANDLE_ERROR( cudaMalloc((void**) &dworld.ptr_objects, world.ptr_objects_dim*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dworld.size_of_objects, world.size_of_objects_dim*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dworld.data_objects, world.data_objects_dim*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dworld.ptr_nodes, world.ptr_nodes_dim*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dworld.size_of_nodes, world.size_of_nodes_dim*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dworld.child_nodes, world.child_nodes_dim*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dworld.mother_node, world.mother_node_dim*sizeof(ui32)) );

    // Copy data to the GPU
    dworld.cur_node_id = world.cur_node_id;
    dworld.ptr_objects_dim = world.ptr_objects_dim;
    dworld.size_of_objects_dim = world.size_of_objects_dim;
    dworld.data_objects_dim = world.data_objects_dim;
    dworld.ptr_nodes_dim = world.ptr_nodes_dim;
    dworld.size_of_nodes_dim = world.size_of_nodes_dim;
    dworld.child_nodes_dim = world.child_nodes_dim;
    dworld.mother_node_dim = world.mother_node_dim;

    HANDLE_ERROR( cudaMemcpy(dworld.ptr_objects, world.ptr_objects,
                             world.ptr_objects_dim*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dworld.size_of_objects, world.size_of_objects,
                             world.size_of_objects_dim*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dworld.data_objects, world.data_objects,
                             world.data_objects_dim*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(dworld.ptr_nodes, world.ptr_nodes,
                             world.ptr_nodes_dim*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dworld.size_of_nodes, world.size_of_nodes,
                             world.size_of_nodes_dim*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dworld.child_nodes, world.child_nodes,
                             world.child_nodes_dim*sizeof(ui32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dworld.mother_node, world.mother_node,
                             world.mother_node_dim*sizeof(ui32), cudaMemcpyHostToDevice) );

}






























#endif
