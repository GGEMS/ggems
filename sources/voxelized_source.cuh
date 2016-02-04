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

#ifndef VOXELIZED_SOURCE_CUH
#define VOXELIZED_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"
#include "prng.cuh"
#include "constants.cuh"
#include "fun.cuh"

//// External function
__host__ __device__ void voxelized_source_primary_generator(ParticleStack particles, ui32 id,
                                                            f32 *cdf_index, f32 *cdf_act, ui32 nb_acts,
                                                            f32 px, f32 py, f32 pz,
                                                            ui32 nb_vox_x, ui32 nb_vox_y, ui32 nb_vox_z,
                                                            f32 sx, f32 sy, f32 sz,
                                                            f32 energy, ui8 type, ui32 geom_id);

__host__ __device__ void voxelized_source_primary_mono_generator(ParticleStack particles, ui32 id,
                                                            f32 *cdf_index, f32 *cdf_act, ui32 nb_acts,
                                                            f32 px, f32 py, f32 pz,
                                                            ui32 nb_vox_x, ui32 nb_vox_y, ui32 nb_vox_z,
                                                            f32 sx, f32 sy, f32 sz,
                                                            f32 energy, ui8 type, ui32 geom_id);


// VoxelizedSource
class VoxelizedSource {
    public:
        VoxelizedSource();

        void set_position(f32 vpx, f32 vpy, f32 vpz);
        void set_energy(f32 venergy);
        void set_histpoint(f32 venergy, f32 vpart);
        void set_source_type(std::string vtype);
        void set_seed(ui32 vseed);
        void set_in_geometry(ui32 vgeometry_id);
        void set_source_name(std::string vsource_name);

        void load_from_mhd(std::string filename);
        void compute_cdf();

        ui32 seed, geometry_id;
        std::string source_name, source_type;
        f32 px, py, pz;
        f32 energy;

        ui16 nb_vox_x, nb_vox_y, nb_vox_z;
        ui32 number_of_voxels;
        f32 spacing_x, spacing_y, spacing_z;

        // Activities
        f32 *activity_volume;
        f32 tot_activity;
        // CDF
        f32 *activity_cdf;
        f32 *activity_index;
        ui32 activity_size;
        
        std::vector<f32> energy_hist, partpdec;

    private:
        // For mhd
        void skip_comment(std::istream &);
        std::string remove_white_space(std::string);
        std::string read_mhd_key(std::string);
        std::string read_mhd_string_arg(std::string);
        i32 read_mhd_int(std::string);
        i32 read_mhd_int_atpos(std::string, i32);
        f32 read_mhd_f32_atpos(std::string, i32);

};

#endif
