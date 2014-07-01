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

MeshedPhantom::MeshedPhantom() {}

// Load a mesh from raw data exported by Blender
void MeshedPhantom::load(std::string filename) {
    std::string line;

    std::ifstream file(filename.c_str());

    // Clear the list of triangles
    triangles.clear();

    if(!file) { printf("Error, file %s not found \n",filename.c_str()); exit(EXIT_FAILURE);}
    while (file) {
        std::getline(file, line);

        if (file) {
            Triangle tri;
            float val;
            std::string txt;
            int pos;

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.u.x = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.u.y = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.u.z = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.v.x = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.v.y = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.v.z = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.w.x = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.w.y = val;
            line = line.substr(pos+1);

            pos = line.find(" ");
            txt = line.substr(0, pos);
            std::stringstream(txt) >> val;
            tri.w.z = val;
            line = line.substr(pos+1);

            // Add a new triangle
            triangles.push_back(tri);

        }
    }

}

// Set the material
void MeshedPhantom::set_material(std::string matname) {
    material_name = matname;
}

// Scaling
void MeshedPhantom::scale(float3 s) {
    // Scale every triangle from the mesh
    int i=0;
    while (i < triangles.size()) {
        triangles[i].u = f3_mul(triangles[i].u, s);
        triangles[i].v = f3_mul(triangles[i].v, s);
        triangles[i].w = f3_mul(triangles[i].w, s);
        ++i;
    }
}

// Scaling
void MeshedPhantom::scale(float sx, float sy, float sz) {
    float3 s = {sx, sy, sz};
    scale(s);
}

// Rotation
void MeshedPhantom::rotate(float3 r) {

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

    // Rotate every triangle from the mesh
    int i=0;
    while(i < triangles.size()) {
        triangles[i].u = m3f3_mul(rot, triangles[i].u);
        triangles[i].v = m3f3_mul(rot, triangles[i].v);
        triangles[i].w = m3f3_mul(rot, triangles[i].w);
        ++i;
    }

}

// Rotation
void MeshedPhantom::rotate(float phi, float theta, float psi) {
    float3 r = {phi, theta, psi};
    rotate(r);
}

// Translation
void MeshedPhantom::translate(float3 t) {
    // Translate every triangle from the mesh
    int i=0;
    while (i < triangles.size()) {
        triangles[i].u = f3_add(triangles[i].u, t);
        triangles[i].v = f3_add(triangles[i].v, t);
        triangles[i].w = f3_add(triangles[i].w, t);
        ++i;
    }
}

// Translation
void MeshedPhantom::translate(float tx, float ty, float tz) {
    float3 t = {tx, ty, tz};
    translate(t);
}





#endif
