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

#ifndef SOURCE_BUILDER_CUH
#define SOURCE_BUILDER_CUH

#include "global.cuh"
#include "point_source.cuh"
#include "cone_beam_source.cuh"
#include "voxelized_source.cuh"

#define POINT_SOURCE 0
#define CONE_BEAM_SOURCE 1

#define ADR_SRC_TYPE 0
#define ADR_SRC_GEOM_ID 1

// Point source
#define ADR_POINT_SRC_PX 2
#define ADR_POINT_SRC_PY 3
#define ADR_POINT_SRC_PZ 4
#define ADR_POINT_SRC_ENERGY 5

// Cone Beam source
#define ADR_CONE_BEAM_SRC_PX 2
#define ADR_CONE_BEAM_SRC_PY 3
#define ADR_CONE_BEAM_SRC_PZ 4
#define ADR_CONE_BEAM_SRC_PHI 5
#define ADR_CONE_BEAM_SRC_THETA 6
#define ADR_CONE_BEAM_SRC_PSI 7
#define ADR_CONE_BEAM_SRC_APERTURE 8
#define ADR_CONE_BEAM_SRC_ENERGY 9

struct Sources {
    // Source structure
    ui32 *ptr_sources;      // Address to access to the different sources
    //ui32 *size_of_sources;  // Size of each source FIXME not need?
    f32 *data_sources;            // Parameters of each source
    ui32 *seeds;            // List of seeds
    ui32 nb_sources;

    // Dimension of each vector
    ui32 ptr_sources_dim;
    //ui32 size_of_sources_dim;
    ui32 data_sources_dim;
    ui32 seeds_dim;
};

// Class to manage sources on the simulation
class SourceBuilder {
    public:
        SourceBuilder();
        void add_source(PointSource src);
        void add_source(ConeBeamSource src);
        void add_source(VoxelizedSource src);

        //void save_ggems_geometry(std::string filename);

        Sources sources;

    private:        



};

#endif
