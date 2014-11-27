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

#ifndef VOXELIZED_CUH
#define VOXELIZED_CUH

#include "global.cuh"
#include "base_object.cuh"

// Voxelized phantom
class Voxelized : public BaseObject {
    public:
        Voxelized();

        void load_from_raw(std::string volume_name, std::string range_name,
                           i32 nx, i32 ny, i32 nz, f32 sx, f32 sy, f32 sz);

        void load_from_mhd(std::string volume_name, std::string range_name);

        void set_object_name(std::string objname);

        f32 *data;

        ui16 nb_vox_x, nb_vox_y, nb_vox_z;
        ui32 number_of_voxels;
        ui32 mem_size; // TODO this can be remove
        f32 spacing_x, spacing_y, spacing_z;

        std::vector<std::string> list_of_materials;

    private:
        void define_materials_from_range(f32 *raw_data, std::string range_name);

        void skip_comment(std::istream &);
        std::string remove_white_space(std::string);

        // for range file
        f32 read_start_range(std::string);
        f32 read_stop_range(std::string);
        std::string read_mat_range(std::string);

        // for mhd
        std::string read_mhd_key(std::string);
        std::string read_mhd_string_arg(std::string);
        i32 read_mhd_int(std::string);
        i32 read_mhd_int_atpos(std::string, i32);
        f32 read_mhd_f32_atpos(std::string, i32);
};




#endif
