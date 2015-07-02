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
#include "cylinder_source.cuh"
#include "planar_source.cuh"

#define POINT_SOURCE 0
#define CONE_BEAM_SOURCE 1
#define VOXELIZED_SOURCE 2
#define CYLINDER_SOURCE 3
#define PLANAR_SOURCE 4

//// Source def
#define ADR_SRC_TYPE 0
#define ADR_SRC_GEOM_ID 1
#define ADR_SRC_NB_PEAK 2
#define ADR_SRC_ENERGY 3
#define ADR_SRC_PARTPDEC 4

// Point source
#define ADR_POINT_SRC_PX 5
#define ADR_POINT_SRC_PY 6
#define ADR_POINT_SRC_PZ 7

// Cylinder source
#define ADR_CYLINDER_SRC_PX 5
#define ADR_CYLINDER_SRC_PY 6
#define ADR_CYLINDER_SRC_PZ 7
#define ADR_CYLINDER_SRC_RAD 8
#define ADR_CYLINDER_SRC_LEN 9

// Planar source
#define ADR_PLANAR_SRC_PX 5
#define ADR_PLANAR_SRC_PY 6
#define ADR_PLANAR_SRC_PZ 7
#define ADR_PLANAR_SRC_WID 8
#define ADR_PLANAR_SRC_LEN 9

// Cone Beam source
#define ADR_CONE_BEAM_SRC_PX 5
#define ADR_CONE_BEAM_SRC_PY 6
#define ADR_CONE_BEAM_SRC_PZ 7
#define ADR_CONE_BEAM_SRC_PHI 8
#define ADR_CONE_BEAM_SRC_THETA 9
#define ADR_CONE_BEAM_SRC_PSI 10
#define ADR_CONE_BEAM_SRC_APERTURE 11
#define ADR_CONE_BEAM_SRC_ENERGY 12

// Voxelized source
#define ADR_VOX_SOURCE_PX 5
#define ADR_VOX_SOURCE_PY 6
#define ADR_VOX_SOURCE_PZ 7
#define ADR_VOX_SOURCE_NB_VOX_X 8
#define ADR_VOX_SOURCE_NB_VOX_Y 9
#define ADR_VOX_SOURCE_NB_VOX_Z 10
#define ADR_VOX_SOURCE_SPACING_X 11
#define ADR_VOX_SOURCE_SPACING_Y 12
#define ADR_VOX_SOURCE_SPACING_Z 13
#define ADR_VOX_SOURCE_ENERGY 14
#define ADR_VOX_SOURCE_TOT_ACTIVITY 15
#define ADR_VOX_SOURCE_NB_CDF 16
#define ADR_VOX_SOURCE_EMISSION_TYPE 17
#define ADR_VOX_SOURCE_CDF_INDEX 18

#define SIZE_POINT_SRC 8
#define SIZE_CYLINDER_SRC 10
#define SIZE_PLANAR_SRC 10

// Emission type
#define EMISSION_MONO 0
#define EMISSION_BACK2BACK 1

struct Sources {
    // Source structure
    ui32 *ptr_sources;        // Address to access to the different sources
    //ui32 *size_of_sources;  // Size of each source FIXME not need?
    f32 *data_sources;        // Parameters of each source
    ui32 *seeds;              // List of seeds
    ui32 nb_sources;

    // Dimension of each vector
    ui32 ptr_sources_dim;
    //ui32 size_of_sources_dim;
    ui32 data_sources_dim;
    ui32 seeds_dim;
};

// External functions
__host__ __device__ void get_primaries(Sources sources, ParticleStack &particles, ui32 id_src, ui32 id_part);
__global__ void kernel_get_primaries(Sources sources, ParticleStack particles, ui32 isrc);


// Class to manage sources on the simulation
class SourceBuilder {
    public:
        SourceBuilder();
        void add_source(PointSource src);
        void add_source(CylinderSource src);
        void add_source(PlanarSource src);
        void add_source(ConeBeamSource src);
        void add_source(VoxelizedSource src);
        
        void set_histpoint(f32 venergy, f32 vpart);

        void copy_source_cpu2gpu();

        //void save_ggems_geometry(std::string filename);

        Sources sources;   // CPU
        Sources dsources;  // GPU

        // Total activity of the source
        f64 tot_activity;

    private:        



};

#endif
