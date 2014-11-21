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

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include <cfloat>
#include "base_object.cuh"

// Voxelized phantom
class Voxelized : public BaseObject {
    public:
        Voxelized();

        void load_from_raw(std::string volume_name, std::string range_name,
                           int nx, int ny, int nz, float sx, float sy, float sz);

        void load_from_mhd(std::string volume_name, std::string range_name);

        void set_object_name(std::string objname);

        float *data;

        unsigned short int nb_vox_x, nb_vox_y, nb_vox_z;
        unsigned int number_of_voxels;
        unsigned int mem_size; // TODO this can be remove
        float spacing_x, spacing_y, spacing_z;

        std::vector<std::string> list_of_materials;

    private:
        void define_materials_from_range(float *raw_data, std::string range_name);

        void skip_comment(std::istream &);
        std::string remove_white_space(std::string);

        // for range file
        float read_start_range(std::string);
        float read_stop_range(std::string);
        std::string read_mat_range(std::string);

        // for mhd
        std::string read_mhd_key(std::string);
        std::string read_mhd_string_arg(std::string);
        int read_mhd_int(std::string);
        int read_mhd_int_atpos(std::string, int);
        float read_mhd_float_atpos(std::string, int);
};




#endif
