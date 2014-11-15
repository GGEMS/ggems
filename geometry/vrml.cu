// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
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

#ifndef VRML_CU
#define VRML_CU

#include "vrml.cuh"

VRML::VRML () {}

//// Private functions ///////////////////////////////////////

// Draw a wireframe AABB
void VRML::draw_wireframe_aabb(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax,
                               Color color, float transparency) {

    //          xmin        xmax
    //          3+---------2+
    //          /          /|
    //         /          / |
    // ymin  0+---------1+  |
    //        |          |  |
    //        | 7+       | 6+   zmax
    //        |          | /
    //        |          |/
    // ymax  4+---------5+   zmin

    fprintf(pfile, "Shape {\n");

//    fprintf(pfile, "  appearance Appearance {\n");
//    fprintf(pfile, "    material Material {\n");
//    fprintf(pfile, "      diffuseColor %f %f %f\n", color.r, color.g, color.b);
//    fprintf(pfile, "      transparency %f\n", transparency);
//    fprintf(pfile, "    }\n");
//    fprintf(pfile, "  }\n");

    fprintf(pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf(pfile, "    coord Coordinate {\n");
    fprintf(pfile, "      point [\n");
    fprintf(pfile, "        %f %f %f,\n", xmin, ymin, zmin); // 0
    fprintf(pfile, "        %f %f %f,\n", xmax, ymin, zmin); // 1
    fprintf(pfile, "        %f %f %f,\n", xmax, ymin, zmax); // 2
    fprintf(pfile, "        %f %f %f,\n", xmin, ymin, zmax); // 3
    fprintf(pfile, "        %f %f %f,\n", xmin, ymax, zmin); // 4
    fprintf(pfile, "        %f %f %f,\n", xmax, ymax, zmin); // 5
    fprintf(pfile, "        %f %f %f,\n", xmax, ymax, zmax); // 6
    fprintf(pfile, "        %f %f %f,\n", xmin, ymax, zmax); // 7
    fprintf(pfile, "      ]\n");
    fprintf(pfile, "    }\n");
    // CoordIndex
    fprintf(pfile, "    coordIndex [\n");
    fprintf(pfile, "      %i, %i, %i, %i, %i, -1,\n", 0, 1, 2, 3, 0); // top
    fprintf(pfile, "      %i, %i, %i, %i, %i, -1,\n", 4, 5, 6, 7, 4); // bottom
    fprintf(pfile, "      %i, %i, -1,\n", 0, 4);
    fprintf(pfile, "      %i, %i, -1,\n", 1, 5);
    fprintf(pfile, "      %i, %i, -1,\n", 2, 6);
    fprintf(pfile, "      %i, %i, -1,\n", 3, 7);
    fprintf(pfile, "    ]\n");

    // Color
    fprintf(pfile, "    color Color {\n");
    fprintf(pfile, "      color [%f %f %f]\n", color.r, color.g, color.b);
    fprintf(pfile, "    }\n");
    fprintf(pfile, "    colorIndex [0, 0, 0, 0, 0, 0]\n");
    fprintf(pfile, "    colorPerVertex FALSE\n");

    fprintf(pfile, "  }\n");
    fprintf(pfile, "}\n");
}

// Open the VRML file
void VRML::open(std::string filename) {
    // check extension
    if (filename.size() < 3) {
        printf("Error, to export a vrml file, the exension must be '.wrl'!\n");
        return;
    }
    std::string ext = filename.substr(filename.size()-3);
    if (ext!="wrl") {
        printf("Error, to export a vrml file, the exension must be '.wrl'!\n");
        return;
    }

    pfile = fopen(filename.c_str(), "w");

    // Write the header
    fprintf(pfile, "#VRML V2.0 utf8\n");
    fprintf(pfile, "# Exported from GGEMS\n");
}

// Draw an AABB
void VRML::draw_aabb(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax,
                     Color color, float transparency) {


    fprintf(pfile, "Transform {\n");
    fprintf(pfile, "  translation %f %f %f\n", xmin+(xmax-xmin)/2.0,
                                               ymin+(ymax-ymin)/2.0,
                                               zmin+(zmax-zmin)/2.0);
    fprintf(pfile, "  children [\n");
    fprintf(pfile, "    Shape {\n");
    fprintf(pfile, "      appearance Appearance {\n");
    fprintf(pfile, "        material Material {\n");
    fprintf(pfile, "          diffuseColor %f %f %f\n", color);
    fprintf(pfile, "          transparency %f\n", transparency);
    fprintf(pfile, "        }\n");
    fprintf(pfile, "      }\n");
    fprintf(pfile, "      geometry Box {\n");
    fprintf(pfile, "        size %f %f %f\n", xmax-xmin, ymax-ymin, zmax-zmin);
    fprintf(pfile, "      }\n");
    fprintf(pfile, "    }\n");
    fprintf(pfile, "  ]\n");
    fprintf(pfile, "}\n");

}

/////////////////////////////////////////////////////////////////////

// Export the geometry /////////////////////////////////////
void VRML::write_geometry(GeometryBuilder geometry) {

    //printf("Nb obj %i\n", geometry.name_objects.size());

    // Read each object contains on the world
    unsigned int iobj = 0;
    while (iobj < geometry.name_objects.size()) {

        //printf("name %s\n", geometry.name_objects[iobj].c_str());

        fprintf(pfile, "\n# %s\n", geometry.name_objects[iobj].c_str());

        // Get the type of the volume
        unsigned int adr_obj = geometry.world.ptr_objects[iobj];
        unsigned int type_obj = geometry.world.data_objects[adr_obj + ADR_OBJ_TYPE];

        //printf("obj type %i\n", type_obj);

        // Draw the volume accordingly
        if (type_obj == AABB) {

            float xmin = geometry.world.data_objects[adr_obj + ADR_AABB_XMIN];
            float xmax = geometry.world.data_objects[adr_obj + ADR_AABB_XMAX];
            float ymin = geometry.world.data_objects[adr_obj + ADR_AABB_YMIN];
            float ymax = geometry.world.data_objects[adr_obj + ADR_AABB_YMAX];
            float zmin = geometry.world.data_objects[adr_obj + ADR_AABB_ZMIN];
            float zmax = geometry.world.data_objects[adr_obj + ADR_AABB_ZMAX];

            if (geometry.object_wireframe[iobj]) {
                draw_wireframe_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                                    geometry.object_colors[iobj],
                                    geometry.object_transparency[iobj]);
            } else {
                draw_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                          geometry.object_colors[iobj],
                          geometry.object_transparency[iobj]);
            }

        } else if (type_obj == SPHERE) {
            // do
        } else if (type_obj == VOXELIZED) {

            float xmin = geometry.world.data_objects[adr_obj + ADR_AABB_XMIN];
            float xmax = geometry.world.data_objects[adr_obj + ADR_AABB_XMAX];
            float ymin = geometry.world.data_objects[adr_obj + ADR_AABB_YMIN];
            float ymax = geometry.world.data_objects[adr_obj + ADR_AABB_YMAX];
            float zmin = geometry.world.data_objects[adr_obj + ADR_AABB_ZMIN];
            float zmax = geometry.world.data_objects[adr_obj + ADR_AABB_ZMAX];

            if (geometry.object_wireframe[iobj]) {
                draw_wireframe_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                                    geometry.object_colors[iobj],
                                    geometry.object_transparency[iobj]);
            } else {
                draw_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                          geometry.object_colors[iobj],
                          geometry.object_transparency[iobj]);
            }

        } else if (type_obj == MESHED) {

            fprintf(pfile, "\n# %s\n", geometry.name_objects[iobj].c_str());
            fprintf(pfile, "Shape {\n");
            fprintf(pfile, "  appearance Appearance {\n");
            fprintf(pfile, "    material Material {\n");
            fprintf(pfile, "      diffuseColor %f %f %f\n", geometry.object_colors[iobj].r,
                    geometry.object_colors[iobj].g,
                    geometry.object_colors[iobj].b);
            fprintf(pfile, "      transparency %f\n", geometry.object_transparency[iobj]);
            fprintf(pfile, "    }\n");
            fprintf(pfile, "  }\n");

            fprintf(pfile, "  geometry IndexedFaceSet {\n");
            fprintf(pfile, "    coord Coordinate {\n");
            fprintf(pfile, "      point [\n");
            unsigned int nb_vertices = geometry.world.data_objects[adr_obj + ADR_MESHED_NB_VERTICES];
            unsigned int adr_data = adr_obj + ADR_MESHED_DATA;
            unsigned int ind;
            unsigned int i=0; while (i < nb_vertices) {
                ind = 3*i;
                fprintf(pfile, "        %f %f %f,\n", geometry.world.data_objects[adr_data+ind],
                                                      geometry.world.data_objects[adr_data+ind+1],
                                                      geometry.world.data_objects[adr_data+ind+2]);
                ++i;
            }
            fprintf(pfile, "      ]\n");
            fprintf(pfile, "    }\n");

            fprintf(pfile, "    coordIndex [\n");
            unsigned int nb_triangles = nb_vertices / 3;
            i=0; while (i < nb_triangles) {
                ind = 3*i;
                fprintf(pfile, "      %i, %i, %i, -1,\n", ind, ind+1, ind+2);
                ++i;
            }
            fprintf(pfile, "    ]\n");

            fprintf(pfile, "  }\n");
            fprintf(pfile, "}\n");

            // Octree viewer
            unsigned int octree_type = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_TYPE];

            if (octree_type == REG_OCTREE) {

                // Read first the bounding box
                float gxmin = geometry.world.data_objects[adr_obj+ADR_AABB_XMIN];
                float gxmax = geometry.world.data_objects[adr_obj+ADR_AABB_XMAX];
                float gymin = geometry.world.data_objects[adr_obj+ADR_AABB_YMIN];
                float gymax = geometry.world.data_objects[adr_obj+ADR_AABB_YMAX];
                float gzmin = geometry.world.data_objects[adr_obj+ADR_AABB_ZMIN];
                float gzmax = geometry.world.data_objects[adr_obj+ADR_AABB_ZMAX];

                draw_wireframe_aabb(gxmin, gxmax, gymin, gymax, gzmin, gzmax,
                                    make_color(0.0, 0.0, 1.0),
                                    1.0);

                unsigned int nx = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_NX];
                unsigned int ny = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_NY];
                unsigned int nz = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_NZ];
                float sx = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_SX];
                float sy = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_SY];
                float sz = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_SZ];

                unsigned int adr_octree = adr_data + nb_vertices*3;

                unsigned int ix, iy, iz;
                iz=0; while (iz < nz) {
                    iy=0; while (iy < ny) {
                        ix=0; while (ix < nx) {

                            // If octree cell not empty, draw it
                            ind = iz*ny*nx + iy*nx + ix;
                            if (geometry.world.data_objects[adr_octree+ind] != 0) {
                                float xmin = ix*sx + gxmin; float xmax = xmin+sx;
                                float ymin = iy*sy + gymin; float ymax = ymin+sy;
                                float zmin = iz*sz + gzmin; float zmax = zmin+sz;
                                draw_wireframe_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                                                    make_color(1.0, 0.0, 0.0),
                                                    1.0);
                            }

                            ++ix;
                        } // nx
                        ++iy;

                    } // ny
                    ++iz;

                } // nz


            } // REG_OCTREE

        } // MESHED

        ++iobj;
    }

}

// Export the sources /////////////////////////////////////
void VRML::write_sources(SourceBuilder sources) {
    unsigned int isrc = 0;
    while (isrc < sources.sources.nb_sources) {

        // Read the address source
        unsigned int adr_src = sources.sources.ptr_sources[isrc];

        // Read the kind of sources
        unsigned int type_src = (unsigned int)(sources.sources.data_sources[adr_src + ADR_SRC_TYPE]);

        // Point Source
        if (type_src == POINT_SOURCE) {
            float px = sources.sources.data_sources[adr_src+ADR_POINT_SRC_PX];
            float py = sources.sources.data_sources[adr_src+ADR_POINT_SRC_PY];
            float pz = sources.sources.data_sources[adr_src+ADR_POINT_SRC_PZ];

            fprintf(pfile, "\n# Source\n");
            fprintf(pfile, "Shape {\n");
            fprintf(pfile, "  geometry PointSet {\n");
            fprintf(pfile, "    coord Coordinate {\n");
            fprintf(pfile, "      point [ %f %f %f ]\n", px, py, pz);
            fprintf(pfile, "    }\n");
            fprintf(pfile, "    color Color {\n");
            fprintf(pfile, "      color [ 1.0 1.0 0.0 ]\n");  // Yellow
            fprintf(pfile, "    }\n");
            fprintf(pfile, "  }\n");
            fprintf(pfile, "}\n");

        } else if (type_src == CONE_BEAM_SOURCE) {
            float px = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PX];
            float py = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PY];
            float pz = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PZ];

            fprintf(pfile, "\n# Source\n");
            fprintf(pfile, "Shape {\n");
            fprintf(pfile, "  geometry PointSet {\n");
            fprintf(pfile, "    coord Coordinate {\n");
            fprintf(pfile, "      point [ %f %f %f ]\n", px, py, pz);
            fprintf(pfile, "    }\n");
            fprintf(pfile, "    color Color {\n");
            fprintf(pfile, "      color [ 1.0 1.0 0.0 ]\n");  // Yellow
            fprintf(pfile, "    }\n");
            fprintf(pfile, "  }\n");
            fprintf(pfile, "}\n\n");

            float phi = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PHI];
            float theta = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_THETA];
            float psi = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PSI];

            // Draw the direction as a vector
            //float3 d = f3_rotate(make_float3(0.0f, 0.0f, 1.0f), make_float3(phi, theta, psi));
            float3 d = make_float3(0.0f, 0.0f, 1.0f);
            d = f3_scale(d, 5.0f);   // FIXME find a way to automatically adjust this value

            fprintf(pfile, "Shape {\n");
            fprintf(pfile, "  geometry IndexedLineSet {\n");
            // Coordinate
            fprintf(pfile, "    coord Coordinate {\n");
            fprintf(pfile, "      point [\n");
            fprintf(pfile, "        %f %f %f,\n", px, py, pz);
            fprintf(pfile, "        %f %f %f,\n", px+d.x, py+d.y, pz+d.z);
            fprintf(pfile, "      ]\n");
            fprintf(pfile, "    }\n");
            // CoordIndex
            fprintf(pfile, "    coordIndex [\n");
            fprintf(pfile, "      %i, %i, -1,\n", 0, 1);
            fprintf(pfile, "    ]\n");
            // Color
            fprintf(pfile, "    color Color {\n");
            fprintf(pfile, "      color [1.0 1.0 0.0]\n");
            fprintf(pfile, "    }\n");
            fprintf(pfile, "    colorIndex [0]\n");
            fprintf(pfile, "    colorPerVertex FALSE\n");
            fprintf(pfile, "  }\n");
            fprintf(pfile, "}\n");

        }

        ++isrc;
    }

}

// Export particles /////////////////////////////////////
void VRML::write_particles(HistoryBuilder history) {

    unsigned int istep;
    unsigned int ip = 0;
    while (ip < history.pname.size()) {

        // for each particle draw the path

        fprintf(pfile, "\n# Particle %i\n", ip);
        fprintf(pfile, "Shape {\n");
        fprintf(pfile, "  geometry IndexedLineSet {\n");
        // Coordinate
        fprintf(pfile, "    coord Coordinate {\n");
        fprintf(pfile, "      point [\n");

        istep=0; while (istep < history.nb_steps[ip]) {
            OneParticleStep astep;
            astep = history.history_data[ip][istep];
            fprintf(pfile, "        %f %f %f,\n", astep.pos.x, astep.pos.y, astep.pos.z);
            ++istep;
        }

        fprintf(pfile, "      ]\n");
        fprintf(pfile, "    }\n");

        // CoordIndex
        fprintf(pfile, "    coordIndex [\n");

        fprintf(pfile, "      %i, ", 0);
        istep=1; while (istep < history.nb_steps[ip]) {
            fprintf(pfile, "%i, ", istep);
            ++istep;
        }

        fprintf(pfile, "-1,\n");
        fprintf(pfile, "    ]\n");

        // Color
        fprintf(pfile, "    color Color {\n");
        if (history.pname[ip] == PHOTON) {
            fprintf(pfile, "      color [1.0 1.0 0.0]\n");
        } else if (history.pname[ip] == ELECTRON) {
            fprintf(pfile, "      color [1.0 0.0 0.0]\n");
        }
        fprintf(pfile, "    }\n");
        fprintf(pfile, "    colorIndex [0]\n");
        fprintf(pfile, "    colorPerVertex FALSE\n");

        fprintf(pfile, "  }\n");
        fprintf(pfile, "}\n");

        ++ip;
    }

}


// Close the VRML file
void VRML::close() {
    fclose(pfile);
}



































#endif
