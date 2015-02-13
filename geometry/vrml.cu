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
void VRML::draw_wireframe_aabb(f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                               Color color, f32 transparency) {

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
void VRML::draw_aabb(f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                     Color color, f32 transparency) {


    fprintf(pfile, "Transform {\n");
    fprintf(pfile, "  translation %f %f %f\n", xmin+(xmax-xmin)/2.0,
                                               ymin+(ymax-ymin)/2.0,
                                               zmin+(zmax-zmin)/2.0);
    fprintf(pfile, "  children [\n");
    fprintf(pfile, "    Shape {\n");
    fprintf(pfile, "      appearance Appearance {\n");
    fprintf(pfile, "        material Material {\n");
    fprintf(pfile, "          diffuseColor %f %f %f\n", color.r, color.g, color.b);
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

// Draw a OBB
void VRML::draw_obb(f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                    f32 cx, f32 cy, f32 cz, f32 angx, f32 angy, f32 angz,
                    Color color, f32 transparency) {

    fprintf(pfile, "Transform {\n");
    fprintf(pfile, "  translation %f %f %f\n", cx, cy, cz);
    fprintf(pfile, "  rotation 1.0 0.0 0.0 %f\n", angx*deg);
    fprintf(pfile, "  rotation 0.0 1.0 0.0 %f\n", angy*deg);
    fprintf(pfile, "  rotation 0.0 0.0 1.0 %f\n", angz*deg);
    fprintf(pfile, "  children [\n");
    fprintf(pfile, "    Shape {\n");
    fprintf(pfile, "      appearance Appearance {\n");
    fprintf(pfile, "        material Material {\n");
    fprintf(pfile, "          diffuseColor %f %f %f\n", color.r, color.g, color.b);
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
    ui32 iobj = 0;
    while (iobj < geometry.name_objects.size()) {

        //printf("name %s\n", geometry.name_objects[iobj].c_str());

        fprintf(pfile, "\n# %s\n", geometry.name_objects[iobj].c_str());

        // Get the type of the volume
        ui32 adr_obj = geometry.world.ptr_objects[iobj];
        ui32 type_obj = geometry.world.data_objects[adr_obj + ADR_OBJ_TYPE];

        //printf("obj type %i\n", type_obj);

        // Draw the volume accordingly
        if (type_obj == AABB) {

            f32 xmin = geometry.world.data_objects[adr_obj + ADR_AABB_XMIN];
            f32 xmax = geometry.world.data_objects[adr_obj + ADR_AABB_XMAX];
            f32 ymin = geometry.world.data_objects[adr_obj + ADR_AABB_YMIN];
            f32 ymax = geometry.world.data_objects[adr_obj + ADR_AABB_YMAX];
            f32 zmin = geometry.world.data_objects[adr_obj + ADR_AABB_ZMIN];
            f32 zmax = geometry.world.data_objects[adr_obj + ADR_AABB_ZMAX];

            if (geometry.object_wireframe[iobj]) {
                draw_wireframe_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                                    geometry.object_colors[iobj],
                                    geometry.object_transparency[iobj]);
            } else {
                draw_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                          geometry.object_colors[iobj],
                          geometry.object_transparency[iobj]);
            }
            
        } else if (type_obj == COLLI) {

            f32 xmin = geometry.world.data_objects[adr_obj + ADR_AABB_XMIN];
            f32 xmax = geometry.world.data_objects[adr_obj + ADR_AABB_XMAX];
            f32 ymin = geometry.world.data_objects[adr_obj + ADR_AABB_YMIN];
            f32 ymax = geometry.world.data_objects[adr_obj + ADR_AABB_YMAX];
            f32 zmin = geometry.world.data_objects[adr_obj + ADR_AABB_ZMIN];
            f32 zmax = geometry.world.data_objects[adr_obj + ADR_AABB_ZMAX];

            if (geometry.object_wireframe[iobj]) {
                draw_wireframe_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                                    geometry.object_colors[iobj],
                                    geometry.object_transparency[iobj]);
            } else {
                draw_aabb(xmin, xmax, ymin, ymax, zmin, zmax,
                          geometry.object_colors[iobj],
                          geometry.object_transparency[iobj]);
            }

        } else if (type_obj == OBB) {

            f32 xmin = geometry.world.data_objects[adr_obj + ADR_AABB_XMIN];
            f32 xmax = geometry.world.data_objects[adr_obj + ADR_AABB_XMAX];
            f32 ymin = geometry.world.data_objects[adr_obj + ADR_AABB_YMIN];
            f32 ymax = geometry.world.data_objects[adr_obj + ADR_AABB_YMAX];
            f32 zmin = geometry.world.data_objects[adr_obj + ADR_AABB_ZMIN];
            f32 zmax = geometry.world.data_objects[adr_obj + ADR_AABB_ZMAX];

            f32 obb_centerx = geometry.world.data_objects[adr_obj+ADR_OBB_CENTER_X];
            f32 obb_centery = geometry.world.data_objects[adr_obj+ADR_OBB_CENTER_Y];
            f32 obb_centerz = geometry.world.data_objects[adr_obj+ADR_OBB_CENTER_Z];

//            f32 ux = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_UX];
//            f32 uy = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_UY];
//            f32 uz = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_UZ];
//            f32 vx = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_VX];
//            f32 vy = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_VY];
//            f32 vz = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_VZ];
//            f32 wx = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_WX];
//            f32 wy = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_WY];
//            f32 wz = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_WZ];

            f32 angx = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_ANGX];
            f32 angy = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_ANGY];
            f32 angz = geometry.world.data_objects[adr_obj+ADR_OBB_FRAME_ANGZ];

            if (geometry.object_wireframe[iobj]) {
                // TODO
            } else {
                draw_obb(xmin, xmax, ymin, ymax, zmin, zmax,
                         obb_centerx, obb_centery, obb_centerz, angx, angy, angz,
                         geometry.object_colors[iobj], geometry.object_transparency[iobj]);
            }

        } else if (type_obj == SPHERE) {
            // TODO
        } else if (type_obj == VOXELIZED) {

            f32 xmin = geometry.world.data_objects[adr_obj + ADR_AABB_XMIN];
            f32 xmax = geometry.world.data_objects[adr_obj + ADR_AABB_XMAX];
            f32 ymin = geometry.world.data_objects[adr_obj + ADR_AABB_YMIN];
            f32 ymax = geometry.world.data_objects[adr_obj + ADR_AABB_YMAX];
            f32 zmin = geometry.world.data_objects[adr_obj + ADR_AABB_ZMIN];
            f32 zmax = geometry.world.data_objects[adr_obj + ADR_AABB_ZMAX];

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
            ui32 nb_vertices = geometry.world.data_objects[adr_obj + ADR_MESHED_NB_VERTICES];
            ui32 adr_data = adr_obj + ADR_MESHED_DATA;
            ui32 ind;
            ui32 i=0; while (i < nb_vertices) {
                ind = 3*i;
                fprintf(pfile, "        %f %f %f,\n", geometry.world.data_objects[adr_data+ind],
                                                      geometry.world.data_objects[adr_data+ind+1],
                                                      geometry.world.data_objects[adr_data+ind+2]);
                ++i;
            }
            fprintf(pfile, "      ]\n");
            fprintf(pfile, "    }\n");

            fprintf(pfile, "    coordIndex [\n");
            ui32 nb_triangles = nb_vertices / 3;
            i=0; while (i < nb_triangles) {
                ind = 3*i;
                fprintf(pfile, "      %i, %i, %i, -1,\n", ind, ind+1, ind+2);
                ++i;
            }
            fprintf(pfile, "    ]\n");

            fprintf(pfile, "  }\n");
            fprintf(pfile, "}\n");

            // Octree viewer
            ui32 octree_type = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_TYPE];

            if (octree_type == REG_OCTREE) {

                // Read first the bounding box
                f32 gxmin = geometry.world.data_objects[adr_obj+ADR_AABB_XMIN];
                f32 gxmax = geometry.world.data_objects[adr_obj+ADR_AABB_XMAX];
                f32 gymin = geometry.world.data_objects[adr_obj+ADR_AABB_YMIN];
                f32 gymax = geometry.world.data_objects[adr_obj+ADR_AABB_YMAX];
                f32 gzmin = geometry.world.data_objects[adr_obj+ADR_AABB_ZMIN];
                f32 gzmax = geometry.world.data_objects[adr_obj+ADR_AABB_ZMAX];

                draw_wireframe_aabb(gxmin, gxmax, gymin, gymax, gzmin, gzmax,
                                    make_color(0.0, 0.0, 1.0),
                                    1.0);

                ui32 nx = (ui32)geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_NX];
                ui32 ny = (ui32)geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_NY];
                ui32 nz = (ui32)geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_NZ];
                f32 sx = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_SX];
                f32 sy = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_SY];
                f32 sz = geometry.world.data_objects[adr_obj+ADR_MESHED_OCTREE_SZ];

                ui32 adr_octree = adr_data + nb_vertices*3;

                ui32 ix, iy, iz;
                iz=0; while (iz < nz) {
                    iy=0; while (iy < ny) {
                        ix=0; while (ix < nx) {

                            // If octree cell not empty, draw it
                            ind = iz*ny*nx + iy*nx + ix;
                            if (geometry.world.data_objects[adr_octree+ind] != 0) {
                                f32 xmin = ix*sx + gxmin; f32 xmax = xmin+sx;
                                f32 ymin = iy*sy + gymin; f32 ymax = ymin+sy;
                                f32 zmin = iz*sz + gzmin; f32 zmax = zmin+sz;
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
    ui32 isrc = 0;
    while (isrc < sources.sources.nb_sources) {

        // Read the address source
        ui32 adr_src = sources.sources.ptr_sources[isrc];

        // Read the kind of sources
        ui32 type_src = (ui32)(sources.sources.data_sources[adr_src + ADR_SRC_TYPE]);

        // Point Source
        if (type_src == POINT_SOURCE) {
            f32 px = sources.sources.data_sources[adr_src+ADR_POINT_SRC_PX];
            f32 py = sources.sources.data_sources[adr_src+ADR_POINT_SRC_PY];
            f32 pz = sources.sources.data_sources[adr_src+ADR_POINT_SRC_PZ];

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
            f32 px = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PX];
            f32 py = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PY];
            f32 pz = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PZ];

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

            //f32 phi = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PHI];
            //f32 theta = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_THETA];
            //f32 psi = sources.sources.data_sources[adr_src+ADR_CONE_BEAM_SRC_PSI];

            // Draw the direction as a vector
            //f32xyz d = f3_rotate(make_f32xyz(0.0f, 0.0f, 1.0f), make_f32xyz(phi, theta, psi));
            f32xyz d = make_f32xyz(0.0f, 0.0f, 1.0f);
            d = fxyz_scale(d, 5.0f);   // FIXME find a way to automatically adjust this value

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

    ui32 istep;
    ui32 ip = 0;
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

// Export singles /////////////////////////////////////
void VRML::write_singles(std::vector<aSingle> singles) {

    ui32 is = 0;
    f32 r=1.0;
    f32 g=0.0;
    f32 b=0.0;

    // for each singles draw a red point
    fprintf(pfile, "\n# Singles\n");
    fprintf(pfile, "Shape {\n");
    fprintf(pfile, "  geometry PointSet {\n");
    fprintf(pfile, "    coord Coordinate {\n");
    fprintf(pfile, "      point [\n");

    is=0; while (is < singles.size()) {
        fprintf(pfile, "        %f %f %f,\n", singles[is].px, singles[is].py, singles[is].pz);
        ++is;
    }

    fprintf(pfile, "      ]\n");
    fprintf(pfile, "    }\n");
    fprintf(pfile, "    color Color {\n");
    fprintf(pfile, "      color [\n");
    is=0; while (is < singles.size()) {
        fprintf(pfile, "        %f %f %f,\n", r, g, b);
        ++is;
    }
    fprintf(pfile, "      ]\n");
    fprintf(pfile, "    }\n");

    fprintf(pfile, "  }\n");
    fprintf(pfile, "}\n");
}


// Show the CT /////////////////////////////////////
void VRML::write_ct(Voxelized vol) {

    fprintf(pfile, "\n# Voxelized Volume with colormap\n");

    // Loop over voxel
    ui32 z, y, x, imat;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;
    f32 id_mat;
    std::string mat_name;

    //printf("Tot mat to show %i\n", vol.show_mat.size());

    // Loop over voxel
    z=0; while (z<vol.nb_vox_z) {
        y=0; while (y<vol.nb_vox_y) {
            x=0; while (x<vol.nb_vox_x) {

                id_mat = vol.data[z*vol.nb_vox_y*vol.nb_vox_x + y*vol.nb_vox_x + x];
                mat_name = vol.list_of_materials[id_mat];

                //printf("val %f name %s\n", id_mat, mat_name.c_str());

                // Loop over assigned color
                imat=0; while (imat < vol.show_mat.size()) {

                    //printf("   -> %s\n", vol.show_mat[imat].c_str());

                    // If in the range
                    if (mat_name == vol.show_mat[imat]) {
                        //printf("      ok\n");
                        // Get voxel position and draw it
                        xmin = x*vol.spacing_x + vol.xmin;
                        xmax = xmin + vol.spacing_x;
                        ymin = y*vol.spacing_y + vol.ymin;
                        ymax = ymin + vol.spacing_y;
                        zmin = z*vol.spacing_z + vol.zmin;
                        zmax = zmin + vol.spacing_z;
                        draw_aabb(xmin, xmax, ymin, ymax, zmin, zmax, vol.show_colors[imat], vol.show_transparencies[imat]);
                    }

                    ++imat;
                } // imap

                ++x;
            } // x

            ++y;
        } // y

        ++z;
    } // z
}




// Close the VRML file
void VRML::close() {
    fclose(pfile);
}



































#endif
