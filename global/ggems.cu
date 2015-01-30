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

#ifndef GGEMS_CU
#define GGEMS_CU

#include "ggems.cuh"

///////// Simulation Builder class ////////////////////////////////////////////////

SimulationBuilder::SimulationBuilder() {
    target = CPU_DEVICE;

    // Init physics list and secondaries list
    parameters.physics_list = (ui8*)malloc(NB_PROCESSES*sizeof(ui8));
    parameters.secondaries_list = (ui8*)malloc(NB_PARTICLES*sizeof(ui8));

    ui32 i = 0;
    while (i < NB_PROCESSES) {
        parameters.physics_list[i] = DISABLED;
        ++i;
    }
    i = 0;
    while (i < NB_PARTICLES) {
        parameters.secondaries_list[i] = DISABLED;
        ++i;
    }

    // Parameters
    parameters.record_dose_flag = DISABLED;
    parameters.digitizer_flag = DISABLED;
    parameters.nb_of_particles = 0;
    parameters.nb_iterations = 0;
    parameters.time = 0;
    parameters.seed = 0;
    parameters.cs_table_nbins = 0;
    parameters.cs_table_min_E = 0;
    parameters.cs_table_max_E = 0;
    history.record_flag = DISABLED;

    // Init by default others parameters
    gpu_id = 0;
    gpu_block_size = 512;

    // Others parameters
    display_run_time_flag = false;
    display_memory_usage_flag = false;

}

////// :: Main functions ::

// Generate particle based on the sources (CPU version)
void SimulationBuilder::primaries_generator() {

    /// CPU ///////////////////////////////////
    if (target == CPU_DEVICE) {
#ifdef DEBUG
        printf("CPU: primaries generator\n");
#endif

        f64 t_start;
        if (display_run_time_flag) t_start = get_time();

        // Loop over particle slot
        ui32 id = 0;
        ui32 is = 0;
        while (id < particles.stack.size) {

            // TODO - Generic and multi-sources
            //      Read CDF sources
            //      Rnd sources
            is = 0; // first source

            // Get a new particle
            get_primaries(sources.sources, particles.stack, is, id);

            // Next particle
            ++id;

        } // id

        // History record (use only for VRML view)
        if (history.record_flag == ENABLED) {
            id=0; while (id < particles.stack.size) {
                // Record the first position for the tracking history
                history.cpu_new_particle_track(PHOTON);
                history.cpu_record_a_step(particles.stack, id);
                ++id;
            }
        }

        if (display_run_time_flag) {
            print_time("Primaries generator", get_time()-t_start);
        }

    /// GPU /////////////////////////////////////
    } else {

#ifdef DEBUG
        printf("GPU: primaries generator\n");
#endif

        cudaEvent_t t_start, t_stop;
        if (display_run_time_flag) {
            cudaEventCreate(&t_start);
            cudaEventCreate(&t_stop);
            cudaEventRecord(t_start);
        }

        // TODO - Generic and multi-sources
        //      Read CDF sources
        //      Rnd sources
        ui32 is = 0; // first source

        // Kernel
        dim3 threads, grid;
        threads.x = gpu_block_size;
        grid.x = (particles.dstack.size + gpu_block_size - 1) / gpu_block_size;
        kernel_get_primaries<<<grid, threads>>>(sources.dsources, particles.dstack, is);
        cuda_error_check("Error ", " Kernel_primaries_generator");

        if (display_run_time_flag) {
            cudaEventRecord(t_stop);
            cudaEventSynchronize(t_stop);
            f32 time_ms = 0;
            cudaEventElapsedTime(&time_ms, t_start, t_stop);
            print_time("Primaries generator", time_ms/1000.0); // in s
        }

    }

}

// Main navigation
void SimulationBuilder::main_navigator() {

    /// CPU ///////////////////////////////
    if (target == CPU_DEVICE) {

#ifdef DEBUG
        printf("CPU: main navigator\n");
#endif

        f64 t_start;
        if (display_run_time_flag) t_start = get_time();

        cpu_main_navigator(particles.stack, geometry.world,
                           materials.materials_table, cs_tables.photon_CS_table, parameters,
                           digitizer.singles, history);

        if (display_run_time_flag) {
            print_time("Main navigation", get_time()-t_start);
        }

    /// GPU ///////////////////////////////
    } else {

#ifdef DEBUG
        printf("GPU: main navigator\n");
#endif

        cudaEvent_t t_start, t_stop;
        if (display_run_time_flag) {
            cudaEventCreate(&t_start);
            cudaEventCreate(&t_stop);
            cudaEventRecord(t_start);
        }

        gpu_main_navigator(particles.dstack, geometry.dworld,
                           materials.dmaterials_table, cs_tables.dphoton_CS_table, dparameters,
                           digitizer.dsingles, gpu_block_size);

        if (display_run_time_flag) {
            cudaEventRecord(t_stop);
            cudaEventSynchronize(t_stop);
            f32 time_ms = 0;
            cudaEventElapsedTime(&time_ms, t_start, t_stop);
            print_time("Main navigation", time_ms/1000.0); // in s
        }
    }

}

// Copy the global simulation parameters to the GPU
void SimulationBuilder::copy_parameters_cpu2gpu() {

    // Mem allocation
    HANDLE_ERROR( cudaMalloc((void**) &dparameters.physics_list, NB_PROCESSES*sizeof(ui8)) );
    HANDLE_ERROR( cudaMalloc((void**) &dparameters.secondaries_list, NB_PARTICLES*sizeof(ui8)) );

    // Copy data
    HANDLE_ERROR( cudaMemcpy(dparameters.physics_list, parameters.physics_list,
                         sizeof(ui8)*NB_PROCESSES, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dparameters.secondaries_list, parameters.secondaries_list,
                         sizeof(ui8)*NB_PARTICLES, cudaMemcpyHostToDevice) );

    dparameters.record_dose_flag = parameters.record_dose_flag;
    dparameters.digitizer_flag = parameters.digitizer_flag;
    dparameters.nb_of_particles = parameters.nb_of_particles;
    dparameters.nb_iterations = parameters.nb_iterations;
    dparameters.time = parameters.time;
    dparameters.seed = parameters.seed;
    dparameters.cs_table_nbins = parameters.cs_table_nbins;
    dparameters.cs_table_min_E = parameters.cs_table_min_E;
    dparameters.cs_table_max_E = parameters.cs_table_max_E;

}

////// :: Setting ::

// Set the digitizer
void SimulationBuilder::set_digitizer(Digitizer dig) {
    digitizer = dig;
    parameters.digitizer_flag = ENABLED;
}

// Set the geometry of the simulation
void SimulationBuilder::set_geometry(GeometryBuilder obj) {
    geometry = obj;
}

// Set the materials definition associated to the geometry
void SimulationBuilder::set_materials(MaterialBuilder tab) {
    materials = tab;
}

// Set the particles stack
void SimulationBuilder::set_particles(ParticleBuilder p) {
    particles = p;
}

// Set the list of sources
void SimulationBuilder::set_sources(SourceBuilder src) {
    sources = src;
}

// Set the hardware used for the simulation CPU or GPU (CPU by default)
void SimulationBuilder::set_hardware_target(std::string value) {
    if (value == "GPU") {
        target = GPU_DEVICE;
    } else {
        target = CPU_DEVICE;
    }
}

// Add a process to the physics list
void SimulationBuilder::set_process(std::string process_name) {

    if (process_name == "Compton") {
        parameters.physics_list[PHOTON_COMPTON] = ENABLED;
        // printf("add Compton\n");
    } else if (process_name == "PhotoElectric") {
        parameters.physics_list[PHOTON_PHOTOELECTRIC] = ENABLED;
        // printf("add photoelectric\n");
    } else if (process_name == "Rayleigh") {
        parameters.physics_list[PHOTON_RAYLEIGH] = ENABLED;
        // printf("add Rayleigh\n");
    } else if (process_name == "eIonisation") {
        parameters.physics_list[ELECTRON_IONISATION] = ENABLED;
        // printf("add photoelectric\n");
    } else if (process_name == "eBremsstrahlung") {
        parameters.physics_list[ELECTRON_BREMSSTRAHLUNG] = ENABLED;
        // printf("add photoelectric\n");
    } else if (process_name == "eMultipleScattering") {
        parameters.physics_list[ELECTRON_MSC] = ENABLED;
        // printf("add photoelectric\n");
    } else {
        print_warning("This process is unknow!!\n");
        printf("     -> %s\n", process_name.c_str());
        exit_simulation();
    }
}

// Set parameters to generate cross sections table
void SimulationBuilder::set_CS_table_nbins(ui32 valbin) {parameters.cs_table_nbins = valbin;}
void SimulationBuilder::set_CS_table_E_min(f32 valE) {parameters.cs_table_min_E = valE;}
void SimulationBuilder::set_CS_table_E_max(f32 valE) {parameters.cs_table_max_E = valE;}

// Enable the simulation of a particular secondary particle
void SimulationBuilder::set_secondary(std::string pname) {

    if (pname == "Photon") {
        parameters.secondaries_list[PHOTON] = ENABLED;
        // printf("add Compton\n");
    } else if (pname == "Electron") {
        parameters.secondaries_list[ELECTRON] = ENABLED;
        // printf("add photoelectric\n");
    } else {
        print_warning("Secondary particle type is unknow!!");
        printf("     -> %s\n", pname.c_str());
        exit_simulation();
    }
}

// Set the number of particles required for the simulation
void SimulationBuilder::set_number_of_particles(ui32 nb) {
    nb_of_particles = nb;
}

// Set the maximum number of iterations (watchdog)
void SimulationBuilder::set_max_number_of_iterations(ui32 nb) {
    max_iteration = nb;
}

// Set to record the history of some particles (only for CPU version)
void SimulationBuilder::set_record_history(ui32 nb_particles) {
    history.record_flag = ENABLED;
    history.max_nb_particles = std::min(nb_particles, nb_of_particles);
    history.stack_size = particles.stack.size;
}

// Set the GPU id
void SimulationBuilder::set_GPU_ID(ui32 valid) {
    gpu_id = valid;
}

// Set the GPU block size
void SimulationBuilder::set_GPU_block_size(ui32 val) {
    gpu_block_size = val;
}

// Display run time
void SimulationBuilder::set_display_run_time() {
    display_run_time_flag = true;
}

// Display memory usage
void SimulationBuilder::set_display_memory_usage() {
    display_memory_usage_flag = true;
}

////// :: Getting ::

ParticleBuilder SimulationBuilder::get_particles() {
    return particles;
}

////// :: Command ::

// Init simualtion
void SimulationBuilder::init_simulation() {

    // Run time
    f64 t_start = 0;
    if (display_run_time_flag) {
        t_start = get_time();
    }

    // Memory usage
    ui32 mem = 0;

    // First compute the number of iterations and the size of a stack // TODO Can be improved - JB
    if (nb_of_particles % particles.stack.size) {
        nb_of_iterations = (nb_of_particles / particles.stack.size) + 1;
    } else {
        nb_of_iterations = nb_of_particles / particles.stack.size;
    }
    particles.stack.size = nb_of_particles / nb_of_iterations;
    nb_of_particles = particles.stack.size * nb_of_iterations;

    /// Init the GPU if need
    if (target == GPU_DEVICE) {
        // Reset device
        reset_gpu_device();

        // Set the gpu id
        set_gpu_device(gpu_id);
    }

    /// Stack handling /////////////////////////////

    // Init CPU stack
    particles.cpu_malloc_stack();
    particles.cpu_init_stack_seed();

    // Mem usage
    if (display_memory_usage_flag) {
        ui32 mem_part = 91*particles.stack.size + 4;
        mem += mem_part;
        print_memory("Particles stack", mem_part);
    }

    // If GPU
    if (target == GPU_DEVICE) {
        particles.gpu_malloc_stack();
        particles.copy_seed_cpu2gpu();
    }

    /// Cross sections /////////////////////////////

    // Init Cross sections and physics table
    cs_tables.build_table(materials.materials_table, parameters);

    // Mem usage
    if (display_memory_usage_flag) {
        ui32 n = cs_tables.photon_CS_table.nb_bins;
        ui32 k = cs_tables.photon_CS_table.nb_mat;
        ui32 mem_cs = 4*n + 12*n*k + 12*n*101 + 16;
        mem += mem_cs;
        print_memory("Cross sections", mem_cs);
    }

    // If GPU
    if (target == GPU_DEVICE) {
        cs_tables.copy_cs_table_cpu2gpu();
    }
    //cs_tables.print();

    /// Copy every data to the GPU ////////////////
    copy_parameters_cpu2gpu();
    geometry.copy_scene_cpu2gpu();
    materials.copy_materials_table_cpu2gpu();
    sources.copy_source_cpu2gpu();

    // Mem usage
    if (display_memory_usage_flag) {
        // Parameters
        ui32 mem_params = NB_PROCESSES+NB_PARTICLES+30;
        mem += mem_params;
        print_memory("Parameters", mem_params);
        // Geometry
        ui32 mem_geom = 4*geometry.world.ptr_objects_dim + 4*geometry.world.size_of_objects_dim +
                4*geometry.world.data_objects_dim + 4*geometry.world.ptr_nodes_dim +
                4*geometry.world.size_of_nodes_dim + 4*geometry.world.child_nodes_dim +
                4*geometry.world.mother_node_dim + 32;
        mem += mem_geom;
        print_memory("Geometry", mem_geom);
        // Materials
        ui32 n = materials.materials_table.nb_materials;
        ui32 k = materials.materials_table.nb_elements_total;
        ui32 mem_mat = 10*k + 80*n + 8;
        mem += mem_mat;
        print_memory("Materials", mem_geom);
        // Sources
        ui32 mem_src = 4*sources.sources.ptr_sources_dim + 4*sources.sources.data_sources_dim +
                4*sources.sources.seeds_dim + 16;
        mem += mem_src;
        print_memory("Sources", mem_src);
    }

    /// Digitizer /////////////////////////////////

    // init Digitizer
    if (parameters.digitizer_flag) {
        digitizer.cpu_init_singles(particles.stack.size);

        if (target == GPU_DEVICE) {
            digitizer.gpu_init_singles(particles.stack.size);
        }

        // Mem usage
        if (display_memory_usage_flag) {
            ui32 mem_singles = 64*digitizer.singles.size + 4;
            mem += mem_singles;
            print_memory("Singles", mem_singles);
        }
    }

    // Run time
    if (display_run_time_flag) {
        print_time("Initialization", get_time()-t_start);
    }

    // Mem usage
    if (display_memory_usage_flag) {
        print_memory("Total memory usage", mem);
    }
}

// Start the simulation
void SimulationBuilder::start_simulation() {

    ui32 iter = 0;

    // Main loop
    while (iter < nb_of_iterations) {

            // If history is required
            if (target == CPU_DEVICE && history.record_flag == ENABLED) history.cur_iter = iter;

            // Sources
            primaries_generator();

            // Navigation
            main_navigator();

            // Process and store singles on CPU
            if (parameters.digitizer_flag) {
                f64 t_start = get_time();

                if (target == GPU_DEVICE) {
                    digitizer.copy_singles_gpu2cpu();
                }

                digitizer.process_singles(iter);
                digitizer.export_singles();

                // Run time
                if (display_run_time_flag) {
                    print_time("Process singles", get_time()-t_start);
                }
            }

        // iter
        ++iter;
        printf(">> Iter %i / %i\n", iter, nb_of_iterations);

    } // main loop

  }


////// :: Utils ::

#endif
