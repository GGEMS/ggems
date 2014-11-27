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

#ifndef YVA_CU
#define YVA_CU

#include "yva.cuh"

YVA::YVA() {}

// Inlcude a meshed object within the voxelized volume
void YVA::include(Meshed obj, unsigned int obj_id) {

    // First store the id of this object
    obj_inc_id = obj_id;

    // Then allocate memory to store overlap information
    overlap_vox = (bool*)malloc(nb_vox_x*nb_vox_y*nb_vox_z*sizeof(bool));

    // Some vars
    f32 vox_xmin, vox_xmax, vox_ymin, vox_ymax, vox_zmin, vox_zmax;
    unsigned int ix, iy, iz, ind;

    // If the mesh has no octree, store overlapping information
    // only for the bounding box
    if (obj.octree_type == NO_OCTREE) {

        // Check every voxel to determine if the mesh overlap one
        // of them
        iz=0; while (iz < nb_vox_z) {
            // Get absolute position of the voxel
            vox_zmin = (iz*spacing_z)+zmin;
            vox_zmax = vox_zmin + spacing_z;

            iy=0; while (iy < nb_vox_y) {
                // Get absolute position of the voxel
                vox_ymin = (iy*spacing_y)+ymin;
                vox_ymax = vox_ymin + spacing_y;

                ix=0; while (ix < nb_vox_x) {
                    // Get absolute position of the voxel
                    vox_xmin = (ix*spacing_x)+xmin;
                    vox_xmax = vox_xmin + spacing_x;

                    // Store if the voxel overalp with the bounding box
                    ind = iz*nb_vox_x*nb_vox_y + iy*nb_vox_x + ix;
                    overlap_vox[ind] = test_AABB_AABB(obj.xmin, obj.xmax, obj.ymin, obj.ymax,
                                       obj.zmin, obj.zmax,
                                       vox_xmin, vox_xmax, vox_ymin, vox_ymax,
                                       vox_zmin, vox_zmax);

                    ++ix;
                } // ix

                ++iy;
            } // iy

            ++iz;
        } // iz

    // If regular octree chech with every non-void cell
    } else if (obj.octree_type == REG_OCTREE) {

        unsigned int cx, cy, cz, cind;
        f32 cell_xmin, cell_xmax, cell_ymin, cell_ymax, cell_zmin, cell_zmax;

        // Check every voxel to determine if the mesh overlap one
        // of them
        iz=0; while (iz < nb_vox_z) {
            // Get absolute position of the voxel
            vox_zmin = (iz*spacing_z)+zmin;
            vox_zmax = vox_zmin + spacing_z;

            iy=0; while (iy < nb_vox_y) {
                // Get absolute position of the voxel
                vox_ymin = (iy*spacing_y)+ymin;
                vox_ymax = vox_ymin + spacing_y;

                ix=0; while (ix < nb_vox_x) {
                    // Get absolute position of the voxel
                    vox_xmin = (ix*spacing_x)+xmin;
                    vox_xmax = vox_xmin + spacing_x;

                    //// Check every non-void octree cell ////////////////

                    // Store if the voxel overalp with the bounding box
                    ind = iz*nb_vox_x*nb_vox_y + iy*nb_vox_x + ix;

                    cz=0; while (cz < obj.nb_cell_z) {
                        // Get absolute position of the cell
                        cell_zmin = (cz*obj.cell_size_z)+obj.zmin;
                        cell_zmax = cell_zmin + obj.cell_size_z;

                        cy=0; while (cy < obj.nb_cell_y) {
                            // Get absolute position of the cell
                            cell_ymin = (cy*obj.cell_size_y)+obj.ymin;
                            cell_ymax = cell_ymin + obj.cell_size_y;

                            cx=0; while (cx < obj.nb_cell_x) {
                                // Get absolute position of the cell
                                cell_xmin = (cx*obj.cell_size_x)+obj.xmin;
                                cell_xmax = cell_xmin + obj.cell_size_x;

                                cind = cz*obj.nb_cell_y*obj.nb_cell_x + cy*obj.nb_cell_x + cx;
                                if (obj.nb_objs_per_cell[cind] != 0) {
                                    overlap_vox[ind] = test_AABB_AABB(cell_xmin, cell_xmax, cell_ymin, cell_ymax,
                                                                      cell_zmin, cell_zmax,
                                                                      vox_xmin, vox_xmax, vox_ymin, vox_ymax,
                                                                      vox_zmin, vox_zmax);
                                }


                                ++cx;
                            } // cx

                            ++cy;
                        } // cy

                        ++cz;
                    } // cz

                    ////////////////////////////////////////////////////

                    ++ix;
                } // ix

                ++iy;
            } // iy

            ++iz;
        } // iz

    }


}

#endif






















