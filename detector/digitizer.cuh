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

#ifndef DIGITIZER_CUH
#define DIGITIZER_CUH

#include "constants.cuh"
#include "global.cuh"

// Struct that handle singles
struct Singles {
    // first pulse
    f32 *pu1_px;
    f32 *pu1_py;
    f32 *pu1_pz;
    f32 *pu1_E;
    f32 *pu1_tof;
    ui32 *pu1_id_part;
    ui32 *pu1_id_geom;
    ui32 *pu1_nb_hits;
    // second pulse
    f32 *pu2_px;
    f32 *pu2_py;
    f32 *pu2_pz;
    f32 *pu2_E;
    f32 *pu2_tof;
    ui32 *pu2_id_part;
    ui32 *pu2_id_geom;
    ui32 *pu2_nb_hits;

    ui32 size;
};

// Digitizer
class Digitizer {
    public:
        Digitizer();
        void init_singles(ui32 nb);
        void set_output_filename(std::string name);
        void process_singles(ui32 iter);
        void export_singles();
        Singles get_singles();

        Singles singles; // Same size than particles stack

    private:
        std::string filename;
        Singles record_singles; // Recorded and processed singles

};

#endif
