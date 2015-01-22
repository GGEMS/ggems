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

#include "geometry_builder.cuh"
#include "source_builder.cuh"
#include "particles.cuh"
#include "singles.cuh"
#include "global.cuh"

// Axis-Aligned Bounding Box
class VRML {
    public:
        VRML();
        void open(std::string filename);
        void write_geometry(GeometryBuilder geometry);
        void write_sources(SourceBuilder sources);
        void write_particles(HistoryBuilder history);
        void write_ct(Voxelized volume);
        void write_singles(Singles singles);
        void close();
    private:
        FILE *pfile;
        void draw_wireframe_aabb(f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                                 Color color, f32 transparency);
        void draw_aabb(f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                       Color color, f32 transparency);
};

#endif
