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

#ifndef GGEMS_CUH
#define GGEMS_CUH

#include "global.cuh"

#include "constants.cuh"
#include "particles.cuh"
#include "cross_sections_builder.cuh"
#include "global.cuh"
#include "geometry_builder.cuh"
#include "materials.cuh"
#include "source_builder.cuh"

#include "aabb.cuh"
#include "obb.cuh"
#include "sphere.cuh"
#include "meshed.cuh"
#include "voxelized.cuh"
#include "point_source.cuh"
#include "digitizer.cuh"

#include "main_navigator.cuh"

#include "vrml.cuh"
#include "mathplot.cuh"
#include "fun.cuh"

#include "flat_panel_detector.cuh"

// Class to manage the hierarchical structure of the world
class SimulationBuilder {
    public:
        SimulationBuilder();

        // Set simulation object
        void set_geometry(GeometryBuilder obj);
        void set_materials(MaterialBuilder tab);
        void set_sources(SourceBuilder src);
        void set_particles(ParticleBuilder p);
        void set_digitizer(Digitizer dig);

        // Setting parameters
        void set_hardware_target(std::string value);
        void set_GPU_ID(ui32 valid);
        void set_GPU_block_size(ui32 val);
        void set_process(std::string process_name);
        void set_secondary(std::string pname);
        void set_number_of_particles(ui32 nb);
        void set_max_number_of_iterations(ui32 nb);
        void set_record_history(ui32 nb_particles);
        void set_CS_table_nbins(ui32 valbin);
        void set_CS_table_E_min(f32 valE);
        void set_CS_table_E_max(f32 valE);

        // Utils
        void set_display_run_time();
        void set_display_memory_usage();

        // Main functions
        void init_simulation();
        void start_simulation();

        // Get data
         ParticleBuilder get_particles();

        // Parameters
        ui16 target;
        ui32 nb_of_particles;
        ui32 nb_of_iterations;
        ui32 max_iteration;

        // Main elements of the simulation
        ParticleBuilder particles;                  // (CPU & GPU)
        GeometryBuilder geometry;                   // (CPU & GPU
        MaterialBuilder materials;                  // (CPU & GPU)
        SourceBuilder sources;
        GlobalSimulationParameters parameters;      // CPU
        GlobalSimulationParameters dparameters;     // GPU
        CrossSectionsBuilder cs_tables;             // (CPU & GPU)
        Digitizer digitizer;                        // (CPU & GPU)

        // Record history for some particles (only CPU version)
        HistoryBuilder history;

    private:

        // Main functions
        void primaries_generator();
        void main_navigator();

        // For GPU
        ui32 gpu_id, gpu_block_size;
        void copy_parameters_cpu2gpu();

        // Parameters
        bool display_run_time_flag, display_memory_usage_flag;

};



#endif
