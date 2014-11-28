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

#include "meshed.cuh"

/////////////////////////////////////////////////////////////////////////////
// Meshed Phantom ///////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

Meshed::Meshed() {
    octree_type = NO_OCTREE;
    number_of_triangles = 0;
    number_of_vertices = 0;
    nb_cell_x = 0;
    nb_cell_y = 0;
    nb_cell_z = 0;
    cell_size_x = 0.0f;
    cell_size_y = 0.0f;
    cell_size_z = 0.0f;
}

// Build a regular octree to improve navigation within meshed phantom
void Meshed::build_regular_octree(ui32 nx, ui32 ny, ui32 nz) {

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
    f32 size_x = xmax-xmin;
    f32 size_y = ymax-ymin;
    f32 size_z = zmax-zmin;

    cell_size_x = size_x / (f32)nx;
    cell_size_y = size_y / (f32)ny;
    cell_size_z = size_z / (f32)nz;

    f32 org_x = xmin;
    f32 org_y = ymin;
    f32 org_z = zmin;

    f32 cell_ix, cell_iy, cell_iz;         // cell position (in voxel ID)
    f32 cell_xmin, cell_ymin, cell_zmin;   // cell position (in 3D space)
    f32 cell_xmax, cell_ymax, cell_zmax;

    f32xyz u, v, w;                          // A triangle
    ui32 addr_tri, itri, ct_tri;
    ui32 cur_cell_index = 0;

    f32 progress = 0.0f;
    printf("\nBuilding regular octree\n");
    printf("progress.... %6.2f %%", progress);
    fflush(stdout);
    f32 inc_progress = 100.0f / f32(nz);

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
                cell_xmax = cell_xmin + cell_size_x;
                cell_ymax = cell_ymin + cell_size_y;
                cell_zmax = cell_zmin + cell_size_z;

                // Search for triangle/AABB collision
                itri = 0;
                ct_tri = 0;
                while (itri < number_of_triangles) {

                    // Define a triangle
                    addr_tri = itri*9;     // 3 vertices xyz
                    u = make_f32xyz(vertices[addr_tri],  vertices[addr_tri+1], vertices[addr_tri+2]);
                    v = make_f32xyz(vertices[addr_tri+3], vertices[addr_tri+4], vertices[addr_tri+5]);
                    w = make_f32xyz(vertices[addr_tri+6], vertices[addr_tri+7], vertices[addr_tri+8]);

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

                cell_xmin += cell_size_x;
                ++cell_ix;

            } // ix

            cell_ymin += cell_size_y;
            ++cell_iy;

        } // iy

        cell_zmin += cell_size_z;
        ++cell_iz;

        // print out
        progress += inc_progress;
        printf("\b\b\b\b\b\b\b\b%6.2f %%", progress);
        fflush(stdout);

    } // iz
    printf("\n");
}


// Load a mesh from raw data exported by Blender
void Meshed::load_from_raw(std::string filename) {

    // In order to define the bounding box of this mesh
    xmin =  FLT_MAX; ymin =  FLT_MAX; zmin =  FLT_MAX;
    xmax = -FLT_MAX; ymax = -FLT_MAX; zmax = -FLT_MAX;

    // To read the file
    std::string line;

    // Get the number of triangles - FIXME: This is uggly!!!!
    std::ifstream file(filename.c_str());
    if(!file) {
        printf("Error, file %s not found \n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    ui32 nb_lines = 0;
    while (file) {
        std::getline(file, line);
        if (file) ++nb_lines;
    }
    file.close();

    // Get the number of triangles (3 vertices xyz)
    number_of_triangles = nb_lines;
    number_of_vertices = number_of_triangles * 3;

    // Allocate data
    vertices = (f32*)malloc(number_of_vertices * 3 * sizeof(f32));

    // Read and load data
    file.open(filename.c_str());
    if (!file) {
        printf("Error, file %s not found \n",filename.c_str());
        exit(EXIT_FAILURE);
    }
    ui32 i = 0.0f;
    while (file) {
        std::getline(file, line);

        if (file) {
            f32 val;
            std::string txt;
            i32 pos;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;            
            vertices[i] = val; // u.x
            ++i;
            line = line.substr(pos+1);
            if (val < xmin) xmin = val; // Bounding box
            if (val > xmax) xmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // u.y
            ++i;
            line = line.substr(pos+1);
            if (val < ymin) ymin = val; // Bounding box
            if (val > ymax) ymax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // u.z
            ++i;
            line = line.substr(pos+1);
            if (val < zmin) zmin = val; // Bounding box
            if (val > zmax) zmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // v.x
            ++i;
            line = line.substr(pos+1);
            if (val < xmin) xmin = val; // Bounding box
            if (val > xmax) xmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // v.y
            ++i;
            line = line.substr(pos+1);
            if (val < ymin) ymin = val; // Bounding box
            if (val > ymax) ymax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // v.z
            ++i;
            line = line.substr(pos+1);
            if (val < zmin) zmin = val; // Bounding box
            if (val > zmax) zmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // w.x
            ++i;
            line = line.substr(pos+1);
            if (val < xmin) xmin = val; // Bounding box
            if (val > xmax) xmax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // w.y
            ++i;
            line = line.substr(pos+1);
            if (val < ymin) ymin = val; // Bounding box
            if (val > ymax) ymax = val;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            vertices[i] = val; // w.z
            ++i;
            line = line.substr(pos+1);
            if (val < zmin) zmin = val; // Bounding box
            if (val > zmax) zmax = val;

        }
    } // while

    // Close the file
    file.close();
}

// Export the mesh and its associated octree to ggems format
void Meshed::save_ggems_mesh(std::string filename) {

    // Check extension
    if (filename.size() < 10) {
        printf("Error, to export a mesh in ggems format, the exension must be '.ggems_mesh'!\n");
        return;
    }
    std::string ext = filename.substr(filename.size()-10);
    if (ext!="ggems_mesh") {
        printf("Error, to export a mesh in ggems format, the exension must be '.ggems_mesh'!\n");
        return;
    }

    // Open the file
    FILE *pfile = fopen(filename.c_str(), "wb");

    // First write the mesh
    ui32 tmp;
    fwrite(&number_of_triangles, 1, sizeof(ui32), pfile);
    fwrite(&number_of_vertices, 1, sizeof(ui32), pfile);

    tmp = object_name.size();
    fwrite(&tmp, 1, sizeof(ui32), pfile);
    fwrite(object_name.c_str(), tmp, sizeof(i8), pfile);

    tmp = material_name.size();
    fwrite(&tmp, 1, sizeof(ui32), pfile);
    fwrite(material_name.c_str(), tmp, sizeof(i8), pfile);

    fwrite(&xmin, 1, sizeof(f32), pfile);
    fwrite(&xmax, 1, sizeof(f32), pfile);
    fwrite(&ymin, 1, sizeof(f32), pfile);
    fwrite(&ymax, 1, sizeof(f32), pfile);
    fwrite(&zmin, 1, sizeof(f32), pfile);
    fwrite(&zmax, 1, sizeof(f32), pfile);

    //               xyz
    fwrite(vertices, 3*number_of_vertices, sizeof(f32), pfile);

    // Then if defined export the associated octree
    fwrite(&octree_type, 1, sizeof(ui16), pfile);
    if (octree_type == REG_OCTREE) {
        fwrite(&nb_cell_x, 1, sizeof(ui32), pfile);
        fwrite(&nb_cell_y, 1, sizeof(ui32), pfile);
        fwrite(&nb_cell_z, 1, sizeof(ui32), pfile);

        fwrite(&cell_size_x, 1, sizeof(f32), pfile);
        fwrite(&cell_size_y, 1, sizeof(f32), pfile);
        fwrite(&cell_size_z, 1, sizeof(f32), pfile);

        tmp = nb_objs_per_cell.size();
        fwrite(&tmp, 1, sizeof(ui32), pfile);
        fwrite(nb_objs_per_cell.data(), tmp, sizeof(f32), pfile);

        tmp = list_objs_per_cell.size();
        fwrite(&tmp, 1, sizeof(ui32), pfile);
        fwrite(list_objs_per_cell.data(), tmp, sizeof(f32), pfile);

        tmp = addr_to_cell.size();
        fwrite(&tmp, 1, sizeof(ui32), pfile);
        fwrite(addr_to_cell.data(), tmp, sizeof(f32), pfile);
    }

    // Close the file
    fclose(pfile);

}

// Load mesh in ggems format
void Meshed::load_from_ggems_mesh(std::string filename) {

    // Check extension
    if (filename.size() < 10) {
        printf("Error, to import a mesh in ggems format, the exension must be '.ggems_mesh'!\n");
        return;
    }
    std::string ext = filename.substr(filename.size()-10);
    if (ext!="ggems_mesh") {
        printf("Error, to import a mesh in ggems format, the exension must be '.ggems_mesh'!\n");
        return;
    }

    // Open the file
    FILE *pfile = fopen(filename.c_str(), "rb");

    // First read the mesh
    ui32 tmp;
    fread(&number_of_triangles, sizeof(ui32), 1, pfile);
    fread(&number_of_vertices, sizeof(ui32), 1, pfile);

    fread(&tmp, sizeof(ui32), 1, pfile);
    object_name.clear();
    object_name.resize(tmp);
    fread(&object_name[0], sizeof(i8), tmp, pfile);

    fread(&tmp, sizeof(ui32), 1, pfile);
    material_name.clear();
    material_name.resize(tmp);
    fread(&material_name[0], sizeof(i8), tmp, pfile);

    fread(&xmin, sizeof(f32), 1, pfile);
    fread(&xmax, sizeof(f32), 1, pfile);
    fread(&ymin, sizeof(f32), 1, pfile);
    fread(&ymax, sizeof(f32), 1, pfile);
    fread(&zmin, sizeof(f32), 1, pfile);
    fread(&zmax, sizeof(f32), 1, pfile);

    free(vertices); //                           xyz
    vertices = (f32*)malloc(number_of_vertices*3 * sizeof(f32));
    fread(vertices, sizeof(f32), 3*number_of_vertices, pfile);

    // Then if defined import the associated octree
    fread(&octree_type, sizeof(ui16), 1, pfile);
    if (octree_type == REG_OCTREE) {
        fread(&nb_cell_x, sizeof(ui32), 1, pfile);
        fread(&nb_cell_y, sizeof(ui32), 1, pfile);
        fread(&nb_cell_z, sizeof(ui32), 1, pfile);

        fread(&cell_size_x, sizeof(f32), 1, pfile);
        fread(&cell_size_y, sizeof(f32), 1, pfile);
        fread(&cell_size_z, sizeof(f32), 1, pfile);

        fread(&tmp, sizeof(ui32), 1, pfile);
        nb_objs_per_cell.clear();
        nb_objs_per_cell.reserve(tmp);
        fread(nb_objs_per_cell.data(), sizeof(f32), tmp, pfile);

        fread(&tmp, sizeof(ui32), 1, pfile);
        list_objs_per_cell.clear();
        list_objs_per_cell.reserve(tmp);
        fwrite(list_objs_per_cell.data(), sizeof(f32), tmp, pfile);

        fread(&tmp, sizeof(ui32), 1, pfile);
        addr_to_cell.clear();
        addr_to_cell.reserve(tmp);
        fwrite(addr_to_cell.data(), sizeof(f32), tmp, pfile);
    }
}

// Scaling
void Meshed::scale(f32xyz s) {
    // Scale every vertex from the mesh
    ui32 i=0;
    while (i < number_of_vertices) {
        ui32 iv = i*3;

        // Create a f32xyz for the current vertex
        f32xyz vertex = make_f32xyz(vertices[iv], vertices[iv+1], vertices[iv+2]);
        // Scale the vertex
        vertex = fxyz_mul(vertex, s);
        // Put back the value into the vertex list
        vertices[iv] = vertex.x; vertices[iv+1] = vertex.y; vertices[iv+2] = vertex.z;

        ++i;
    }
}

// Scaling
void Meshed::scale(f32 sx, f32 sy, f32 sz) {
    f32xyz s = {sx, sy, sz};
    scale(s);
}

// Rotation
void Meshed::rotate(f32xyz r) {

    // f32 deg = pi / 180.0f;
    f32 phi = r.x*deg; // deg is defined by G4 unit system
    f32 theta = r.y*deg;
    f32 psi = r.z*deg;

    f32 sph = sin(phi);
    f32 cph = cos(phi);
    f32 sth = sin(theta);
    f32 cth = cos(theta);
    f32 sps = sin(psi);
    f32 cps = cos(psi);

    // Build rotation matrix
    f32matrix33 rot = { cph*cps-sph*cth*sps,  cph*sps+sph*cth*cps,  sth*sph,
                       -sph*cps-cph*cth*sps, -sph*sps+cph*cth*cps,  sth*cph,
                        sth*sps,             -sth*cps,                  cth};

    // Rotate every vertex from the mesh
    ui32 i=0;
    while (i < number_of_vertices) {
        ui32 iv = i*3;

        // Create a f32xyz for the current vertex
        f32xyz vertex = make_f32xyz(vertices[iv], vertices[iv+1], vertices[iv+2]);
        // Rotate the vertex
        vertex = fmatrixfxyz_mul(rot, vertex);
        // Put back the value into the vertex list
        vertices[iv] = vertex.x; vertices[iv+1] = vertex.y; vertices[iv+2] = vertex.z;

        ++i;
    }

}

// Rotation
void Meshed::rotate(f32 phi, f32 theta, f32 psi) {
    f32xyz r = {phi, theta, psi};
    rotate(r);
}

// Translation
void Meshed::translate(f32xyz t) {

    // Translate every vertex from the mesh
    ui32 i=0;
    while (i < number_of_vertices) {
        ui32 iv = i*3;

        // Create a f32xyz for the current vertex
        f32xyz vertex = make_f32xyz(vertices[iv], vertices[iv+1], vertices[iv+2]);
        // Translate the vertex
        vertex = fxyz_add(vertex, t);
        // Put back the value into the vertex list
        vertices[iv] = vertex.x; vertices[iv+1] = vertex.y; vertices[iv+2] = vertex.z;

        ++i;
    }

}

// Translation
void Meshed::translate(f32 tx, f32 ty, f32 tz) {
    f32xyz t = {tx, ty, tz};
    translate(t);
}


#endif
