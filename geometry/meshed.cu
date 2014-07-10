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

#ifndef MESHED_CU
#define MESHED_CU

#include "meshed.h"

/////////////////////////////////////////////////////////////////////////////
// Meshed Phantom ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

Meshed::Meshed() {
    octree_type = NO_OCTREE;
}

// Build a regular octree to improve navigation within meshed phantom
void Meshed::build_regular_octree(unsigned int nx, unsigned int ny, unsigned int nz) {

    // First check if there is a mesh loaded
    if (number_of_triangles == 0) {
        printf("Before building an octree, you need to load a meshed phantom!!!!\n");
        return;
    }

    // Store the size of this octree and the type
    nb_cell_x = nx;
    nb_cell_y = ny;
    nb_cell_z = nz;
    octree_type = REG_OCTREE;

    // Compute the size of each octree cell
    float size_x = xmax-xmin;
    float size_y = ymax-ymin;
    float size_z = zmax-zmin;

    float spacing_x = size_x / (float)nx;
    float spacing_y = size_y / (float)ny;
    float spacing_z = size_z / (float)nz;

    float org_x = xmin;
    float org_y = ymin;
    float org_z = zmin;


    float cell_ix, cell_iy, cell_iz;         // cell position (in voxel ID)
    float cell_xmin, cell_ymin, cell_zmin;   // cell position (in 3D space)
    float cell_xmax, cell_ymax, cell_zmax;

    float3 u, v, w;                          // A triangle
    unsigned int addr_tri, itri, ct_tri;
    unsigned int cur_cell_index = 0;

    float progress = 0.0f;
    printf("\nBuilding BVH\n");
    printf("progress.... %6.2f %%", progress);
    fflush(stdout);
    float inc_progress = 100.0f / float(nz);

    // For each octree cell
    cell_iz = 0;
    cell_zmin = org_z;
    while(cell_iz < nz) {
        cell_iy = 0;
        cell_ymin = org_y;
        while(cell_iy < ny) {
            cell_ix = 0;
            cell_xmin = org_x;
            while(cell_ix < nx) {

                // -----------------------------------------------------------

                // Define a voxel as AABB primitive
                cell_xmax = cell_xmin + spacing_x;
                cell_ymax = cell_ymin + spacing_y;
                cell_zmax = cell_zmin + spacing_z;

                // Search for triangle/AABB collision
                itri = 0;
                ct_tri = 0;
                while (itri < number_of_triangles) {

                    // Define a triangle
                    addr_tri = itri*9;     // 3 vertices xyz
                    u = make_float3(vertices[addr_tri],  vertices[addr_tri+1], vertices[addr_tri+2]);
                    v = make_float3(vertices[addr_tri+3], vertices[addr_tri+4], vertices[addr_tri+5]);
                    w = make_float3(vertices[addr_tri+6], vertices[addr_tri+7], vertices[addr_tri+8]);

                    // Check if this triangle is overlappin this octree cell
                    if (overlap_AABB_triangle(cell_xmin, cell_xmax, cell_ymin, cell_ymax, cell_zmin, cell_zmax,
                                              u, v, w) != -1) {

                        // Save triangle index and count the total number of triangles per cell
                        list_objs_per_cell.push_back(itri);
                        ++ct_tri;
                    }
                    ++itri;
                }

                // Save the number of objs per leaf and the address to acces to this cell
                nb_objs_per_cell.push_back(ct_tri);

                if (ct_tri != 0) {
                    addr_to_cell.push_back(cur_cell_index);
                } else {
                    addr_to_cell.push_back(-1);
                }
                cur_cell_index += ct_tri;

                // -----------------------------------------------------------

                cell_xmin += spacing_x;
                ++cell_ix;

            } // ix

            cell_ymin += spacing_y;
            ++cell_iy;

        } // iy

        cell_zmin += spacing_z;
        ++cell_iz;

        // print out
        progress += inc_progress;
        printf("\b\b\b\b\b\b\b\b%6.2f %%", progress);
        fflush(stdout);

    } // iz
    printf("\n");


}



// Load a mesh from raw data exported by Blender
void Meshed::load(std::string filename) {

    // In order to define the bounding box of this mesh
    xmin = FLT_MAX; ymin = FLT_MAX; zmin = FLT_MAX;
    xmax = FLT_MIN; ymax = FLT_MIN; zmax = FLT_MIN;

    // To read the file
    std::string line;
    std::ifstream file(filename.c_str());

    // Clear the list of triangles
    vertices.clear();

    if(!file) { printf("Error, file %s not found \n",filename.c_str()); exit(EXIT_FAILURE);}
    while (file) {
        std::getline(file, line);

        if (file) {
            float val;
            std::string txt;
            int pos;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;            
            vertices.push_back(val); // u.x
            line = line.substr(pos+1);
            if (val < xmin) xmin = val; // Bounding box
            if (val > xmax) xmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // u.y
            line = line.substr(pos+1);
            if (val < ymin) ymin = val; // Bounding box
            if (val > ymax) ymax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // u.z
            line = line.substr(pos+1);
            if (val < zmin) zmin = val; // Bounding box
            if (val > zmax) zmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // v.x
            line = line.substr(pos+1);
            if (val < xmin) xmin = val; // Bounding box
            if (val > xmax) xmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // v.y
            line = line.substr(pos+1);
            if (val < ymin) ymin = val; // Bounding box
            if (val > ymax) ymax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // v.z
            line = line.substr(pos+1);
            if (val < zmin) zmin = val; // Bounding box
            if (val > zmax) zmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // w.x
            line = line.substr(pos+1);
            if (val < xmin) xmin = val; // Bounding box
            if (val > xmax) xmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // w.y
            line = line.substr(pos+1);
            if (val < ymin) ymin = val; // Bounding box
            if (val > ymax) ymax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices.push_back(val); // w.z
            line = line.substr(pos+1);
            if (val < zmin) zmin = val; // Bounding box
            if (val > zmax) zmax = val;

        }
    } // while

    // Get the number of triangles (3 vertices xyz)
    number_of_triangles = (vertices.size() / 9);
    number_of_vertices = (vertices.size() / 3);

}

// Set the material
void Meshed::set_material(std::string matname) {
    material_name = matname;
}

// Give a name to this object
void Meshed::set_object_name(std::string objname) {
    object_name = objname;
}


// Scaling
void Meshed::scale(float3 s) {
    // Scale every vertex from the mesh
    unsigned int i=0;
    while (i < number_of_vertices) {
        unsigned int iv = i*3;

        // Create a float3 for the current vertex
        float3 vertex = make_float3(vertices[iv], vertices[iv+1], vertices[iv+2]);
        // Scale the vertex
        vertex = f3_mul(vertex, s);
        // Put back the value into the vertex list
        vertices[iv] = vertex.x; vertices[iv+1] = vertex.y; vertices[iv+2] = vertex.z;

        ++i;
    }
}

// Scaling
void Meshed::scale(float sx, float sy, float sz) {
    float3 s = {sx, sy, sz};
    scale(s);
}

// Rotation
void Meshed::rotate(float3 r) {

    float deg = pi / 180.0f;
    float phi = r.x*deg;
    float theta = r.y*deg;
    float psi = r.z*deg;

    float sph = sin(phi);
    float cph = cos(phi);
    float sth = sin(theta);
    float cth = cos(theta);
    float sps = sin(psi);
    float cps = cos(psi);

    // Build rotation matrix
    matrix3 rot = { cph*cps-sph*cth*sps,  cph*sps+sph*cth*cps,  sth*sph,
                   -sph*cps-cph*cth*sps, -sph*sps+cph*cth*cps,  sth*cph,
                    sth*sps,             -sth*cps,                  cth};

    // Rotate every vertex from the mesh
    unsigned int i=0;
    while (i < number_of_vertices) {
        unsigned int iv = i*3;

        // Create a float3 for the current vertex
        float3 vertex = make_float3(vertices[iv], vertices[iv+1], vertices[iv+2]);
        // Rotate the vertex
        vertex = m3f3_mul(rot, vertex);
        // Put back the value into the vertex list
        vertices[iv] = vertex.x; vertices[iv+1] = vertex.y; vertices[iv+2] = vertex.z;

        ++i;
    }

}

// Rotation
void Meshed::rotate(float phi, float theta, float psi) {
    float3 r = {phi, theta, psi};
    rotate(r);
}

// Translation
void Meshed::translate(float3 t) {

    // Translate every vertex from the mesh
    unsigned int i=0;
    while (i < number_of_vertices) {
        unsigned int iv = i*3;

        // Create a float3 for the current vertex
        float3 vertex = make_float3(vertices[iv], vertices[iv+1], vertices[iv+2]);
        // Translate the vertex
        vertex = f3_add(vertex, t);
        // Put back the value into the vertex list
        vertices[iv] = vertex.x; vertices[iv+1] = vertex.y; vertices[iv+2] = vertex.z;

        ++i;
    }

}

// Translation
void Meshed::translate(float tx, float ty, float tz) {
    float3 t = {tx, ty, tz};
    translate(t);
}





#endif
