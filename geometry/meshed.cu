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

Meshed::Meshed() {}

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
