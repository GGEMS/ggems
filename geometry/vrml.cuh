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

#ifndef VRML_CUH
#define VRML_CUH

#include <stdlib.h>
#include <stdio.h>
#include <string>

#include "geometry_builder.cuh"
#include "source_builder.cuh"
#include "particles.cuh"
#include "global.cuh"

// Axis-Aligned Bounding Box
class VRML {
    public:
        VRML();
        void open(std::string filename);
        void write_geometry(GeometryBuilder geometry);
        void write_sources(SourceBuilder sources);
        void write_particles(HistoryBuilder history);
        void close();
    private:
        FILE *pfile;
        void draw_wireframe_aabb(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax,
                                 Color color, float transparency);
        void draw_aabb(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax,
                       Color color, float transparency);
};

#endif
