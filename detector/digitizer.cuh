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
#include "geometry_builder.cuh"
#include "raytracing.cuh"

#include "Randomize.hh"

// Struct that handle pulses on CPU and GPU
struct Pulses {
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

// Struct that handle a single
struct aSingle {
    f32 px;
    f32 py;
    f32 pz;
    f32 E;
    f32 tof;
    ui32 id_part;
    ui32 id_geom;
    f64 time;
};

// Struct that handles a coincidence
struct aCoincidence {
    // first singles
    f32 s1_px;
    f32 s1_py;
    f32 s1_pz;
    f32 s1_E;
    f32 s1_tof;
    ui32 s1_id_part;
    ui32 s1_id_geom;
    // second singles
    f32 s2_px;
    f32 s2_py;
    f32 s2_pz;
    f32 s2_E;
    f32 s2_tof;
    ui32 s2_id_part;
    ui32 s2_id_geom;
};

// Digitizer
class Digitizer {
    public:
        Digitizer();

        void set_output_singles(std::string name);
        void set_output_coincidences(std::string name);
        void set_output_projection(std::string name, ui32 volid,
                                   f32 xmin, f32 xmax,
                                   f32 ymin, f32 ymax,
                                   f32 zmin, f32 zmax,
                                   f32 sx, f32 sy, f32 sz);
        
        void set_spect_projections(std::string name,
                                   f32 xmin, f32 xmax,
                                   f32 ymin, f32 ymax,
                                   f32 zmin, f32 zmax,
                                   f32 sx, f32 sy, f32 sz);
        
        void set_number_of_projections(ui32 nb_head);
        
        void set_run(ui32 id_run);

        void set_spatial_blurring(f32 SP_res);
        
        void set_energy_blurring(std::string law, f32 E_res, f32 E_ref, f32 E_slope);
        
        void set_energy_window(f32 vE_low, f32 vE_high);
        void set_time_window(f32 vwin_time);

        // Main function
        void process_chain(ui32 iter, f64 tot_activity, Scene geometry);

        // Singles
        void process_singles(ui32 iter, f64 tot_activity);
        void export_singles();
        std::vector<aSingle> get_singles();

        // Coincidences
        void process_coincidences();
        //void export_coincidences(); // TODO
        // get_coincidences // TODO

        // Projection
        void init_projection();
        void process_projection();
        void process_spect_projections(Scene geometry);
        void export_projection();
        void export_spect_projections(ui32 nb_proj, ui32 id_run);

        void cpu_init_pulses(ui32 nb);
        void gpu_init_pulses(ui32 nb);
        void copy_pulses_gpu2cpu();
        void clear_gpu_pulses();
        
        Pulses pulses;  // CPU - Same size than particles stack
        Pulses dpulses; // GPU
        
        // Process chain flag
        bool flag_singles;
        bool flag_coincidences;
        bool flag_projection;
        bool flag_spect_proj;
        
        bool flag_sp_blurring;
        bool flag_energy_blurring;
        
        // Projection format
        bool flag_projXY;
        bool flag_projYZ;
        bool flag_projXZ;

    private:
        std::string singles_filename;
        std::string coincidences_filename;

        std::vector<aSingle> singles; // Recorded and processed singles
        std::vector<aCoincidence> coincidences;

        std::string projection_filename;
        ui32 projection_idvol, projection_nx, projection_ny, projection_nz;
        ui32 projection_run;
        f32 projection_sx, projection_sy, projection_sz;
        f32 projection_xmin, projection_ymin, projection_zmin;
        typedef std::vector<ui32> single_proj;
        std::vector<single_proj> projection;

        ui32 nb_proj;
       
        ui32 index_run;
        
        // Spatial blurring parameters
        f32 SP_res;
        
        // Energy blurring parameters
        std::string law_name;
        f32 E_res, E_ref, E_slope;
        
        // for coincidences
        f32 E_low, E_high, win_time;

        // keep tracking the time
        f64 global_time;
};

#endif
