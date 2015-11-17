// GGEMS Copyright (C) 2015

/*!
 * \file ggems.cuh
 * \brief Main header of GGEMS lib
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Header of the main GGEMS lib
 *
 */

#ifndef GGEMS_CU
#define GGEMS_CU

#include "ggems.cuh"

////// :: GGEMS Const/Dest ::

GGEMS::GGEMS() {

    // Init physics list and secondaries list
    m_parameters_h.physics_list = (bool*)malloc(NB_PROCESSES*sizeof(bool));
    m_parameters_h.secondaries_list = (bool*)malloc(NB_PARTICLES*sizeof(bool));
    
    ui32 i = 0;
    while (i < NB_PROCESSES) {
        m_parameters_h.physics_list[i] = DISABLED;
        ++i;
    }
    i = 0;
    while (i < NB_PARTICLES) {
        m_parameters_h.secondaries_list[i] = DISABLED;
        ++i;
    }

    // Parameters
    m_parameters_h.nb_of_particles = 0;
    m_parameters_h.size_of_particles_batch = 0;
    m_parameters_h.nb_of_batches = 0;
    m_parameters_h.time = 0;
    m_parameters_h.seed = 0;
    m_parameters_h.cs_table_nbins = 0;
    m_parameters_h.cs_table_min_E = 0;
    m_parameters_h.cs_table_max_E = 0;

    // Init by default others parameters
    m_parameters_h.device_target = CPU_DEVICE;
    m_parameters_h.gpu_id = 0;
    m_parameters_h.gpu_block_size = 512;

    // Others parameters
    m_parameters_h.display_run_time = DISABLED;
    m_parameters_h.display_memory_usage = DISABLED;

}

GGEMS::~GGEMS() {
    //delete m_parameters_h;
    //delete m_source;
}

////// :: Setting ::

/// Params

// Set the hardware used for the simulation CPU or GPU (CPU by default)
void GGEMS::set_hardware_target(std::string value) {
    if (value == "GPU") {
        m_parameters_h.device_target = GPU_DEVICE;
    } else {
        m_parameters_h.device_target = CPU_DEVICE;
    }
}

// Set the GPU id
void GGEMS::set_GPU_ID(ui32 valid) {
    m_parameters_h.gpu_id = valid;
}

// Set the GPU block size
void GGEMS::set_GPU_block_size(ui32 val) {
    m_parameters_h.gpu_block_size = val;
}

// Add a process to the physics list
void GGEMS::set_process(std::string process_name) {

    if (process_name == "Compton") {
        m_parameters_h.physics_list[PHOTON_COMPTON] = ENABLED;

    } else if (process_name == "PhotoElectric") {
        m_parameters_h.physics_list[PHOTON_PHOTOELECTRIC] = ENABLED;

    } else if (process_name == "Rayleigh") {
        m_parameters_h.physics_list[PHOTON_RAYLEIGH] = ENABLED;

    } else if (process_name == "eIonisation") {
        m_parameters_h.physics_list[ELECTRON_IONISATION] = ENABLED;

    } else if (process_name == "eBremsstrahlung") {
        m_parameters_h.physics_list[ELECTRON_BREMSSTRAHLUNG] = ENABLED;

    } else if (process_name == "eMultipleScattering") {
        m_parameters_h.physics_list[ELECTRON_MSC] = ENABLED;

    } else {
        print_warning("This process is unknown!!\n");
        printf("     -> %s\n", process_name.c_str());
        exit_simulation();
    }
}

// Enable the simulation of a particular secondary particle
void GGEMS::set_secondary(std::string pname) {

    if (pname == "Photon") {
        m_parameters_h.secondaries_list[PHOTON] = ENABLED;
    } else if (pname == "Electron") {
        m_parameters_h.secondaries_list[ELECTRON] = ENABLED;
    } else {
        print_warning("Secondary particle type is unknow!!");
        printf("     -> %s\n", pname.c_str());
        exit_simulation();
    }
}

// Set the number of particles required for the simulation
void GGEMS::set_number_of_particles(ui64 nb) {
    m_parameters_h.nb_of_particles = nb;
}

// Set the size of particles batch
void GGEMS::set_size_of_particles_batch(ui64 nb) {
    m_parameters_h.size_of_particles_batch = nb;
}

// Set parameters to generate cross sections table
void GGEMS::set_CS_table_nbins(ui32 valbin) {m_parameters_h.cs_table_nbins = valbin;}
void GGEMS::set_CS_table_E_min(f32 valE) {m_parameters_h.cs_table_min_E = valE;}
void GGEMS::set_CS_table_E_max(f32 valE) {m_parameters_h.cs_table_max_E = valE;}

// Set the seed number
void GGEMS::set_seed(ui32 vseed) {
    m_parameters_h.seed = vseed;
}

/// Sources
void GGEMS::set_source(PointSource *aSource) {
    m_sources.set_source(aSource);
}

/// Utils

// Display run time
void GGEMS::set_display_run_time() {
    m_parameters_h.display_run_time = ENABLED;
}

// Display memory usage
void GGEMS::set_display_memory_usage() {
    m_parameters_h.display_memory_usage = ENABLED;
}

////// :: Private functions ::

// Copy the global simulation parameters to the GPU
void GGEMS::m_copy_parameters_cpu2gpu() {

    // Mem allocation
    HANDLE_ERROR( cudaMalloc((void**) &m_parameters_d.physics_list, NB_PROCESSES*sizeof(bool)) );
    HANDLE_ERROR( cudaMalloc((void**) &m_parameters_d.secondaries_list, NB_PARTICLES*sizeof(bool)) );

    // Copy data
    HANDLE_ERROR( cudaMemcpy(m_parameters_d.physics_list, m_parameters_h.physics_list,
                         sizeof(bool)*NB_PROCESSES, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(m_parameters_d.secondaries_list, m_parameters_h.secondaries_list,
                         sizeof(bool)*NB_PARTICLES, cudaMemcpyHostToDevice) );

    m_parameters_d.nb_of_particles = m_parameters_h.nb_of_particles;
    m_parameters_d.size_of_particles_batch = m_parameters_h.size_of_particles_batch;
    m_parameters_d.nb_of_batches = m_parameters_h.nb_of_batches;

    m_parameters_d.device_target = m_parameters_h.device_target;
    m_parameters_d.gpu_id = m_parameters_h.gpu_id;
    m_parameters_d.gpu_block_size = m_parameters_h.gpu_block_size;

    m_parameters_d.time = m_parameters_h.time;
    m_parameters_d.seed = m_parameters_h.seed;

    m_parameters_d.display_run_time = m_parameters_h.display_run_time;
    m_parameters_d.display_memory_usage = m_parameters_h.display_memory_usage;

    m_parameters_d.cs_table_nbins = m_parameters_h.cs_table_nbins;
    m_parameters_d.cs_table_min_E = m_parameters_h.cs_table_min_E;
    m_parameters_d.cs_table_max_E = m_parameters_h.cs_table_max_E;
}

////// :: Main functions ::

// Init simualtion
void GGEMS::init_simulation() {

    // Run time
    f64 t_start = 0;
    if (m_parameters_h.display_run_time) {
        t_start = get_time();
    }

    // Memory usage
    ui32 mem = 0;

    // CPU PRNG
    srand(m_parameters_h.seed);

    // Get the number of batch required
    m_parameters_h.nb_of_batches = m_parameters_h.nb_of_particles / m_parameters_h.size_of_particles_batch;

    // Init the GPU if need
    if (m_parameters_h.device_target == GPU_DEVICE) {
        // Reset device
        reset_gpu_device();

        // Set the gpu id
        set_gpu_device(m_parameters_h.gpu_id);
    }

    // Copy global params to GPU
    if (m_parameters_h.device_target == GPU_DEVICE) {
        m_copy_parameters_cpu2gpu();
    }

    /// Init Sources ///////////////
    m_sources.initialize(m_parameters_h);

    /// Material handling ////////////////////////////

    // Load data
    // TODO

    // Build data based on geometry
    // TODO

    // Copy data to GPU
    if (m_parameters_h.device_target == GPU_DEVICE) {
        m_materials.copy_materials_table_cpu2gpu();
    }

    /// Stack handling ///////////////////////////////

    // Init stack params
    m_particles.set_stack_size(m_parameters_h.size_of_particles_batch);
    m_particles.set_seed(m_parameters_h.seed);
    // Init stack (CPU)
    m_particles.cpu_malloc_stack();
    m_particles.cpu_init_stack_seed();
    // If GPU, init
    if (m_parameters_h.device_target == GPU_DEVICE) {
        m_particles.gpu_malloc_stack();
        m_particles.copy_seed_cpu2gpu();
    }

    // Mem usage
    if (m_parameters_h.display_memory_usage) {
        ui32 mem_part = 91*m_particles.stack_h.size + 4;
        mem += mem_part;
        print_memory("Particles stack", mem_part);
    }

    /// Cross sections /////////////////////////////

    // Init Cross sections and physics table
    m_cross_sections.build_table(m_materials.mat_table_h, m_parameters_h);

    // Mem usage
    if (m_parameters_h.display_memory_usage) {
        ui32 n = m_cross_sections.photon_CS_table_h.nb_bins;
        ui32 k = m_cross_sections.photon_CS_table_h.nb_mat;
        ui32 mem_cs = 4*n + 12*n*k + 12*n*101 + 16;
        mem += mem_cs;
        print_memory("Cross sections", mem_cs);

        // Add CS from others particles
    }

    // If GPU
    if (m_parameters_h.device_target == GPU_DEVICE) {
        m_cross_sections.copy_cs_table_cpu2gpu();
    }
    //cs_tables.print();

    // Mem usage
    if (m_parameters_h.display_memory_usage) {
        // Parameters
        ui32 mem_params = NB_PROCESSES+NB_PARTICLES+30;
        mem += mem_params;
        print_memory("Parameters", mem_params);

        /*
        // Geometry
        ui32 mem_geom = 4*geometry.world.ptr_objects_dim + 4*geometry.world.size_of_objects_dim +
                4*geometry.world.data_objects_dim + 4*geometry.world.ptr_nodes_dim +
                4*geometry.world.size_of_nodes_dim + 4*geometry.world.child_nodes_dim +
                4*geometry.world.mother_node_dim + 32;
        mem += mem_geom;
        print_memory("Geometry", mem_geom);
        */

        // Materials
        ui32 n = m_materials.mat_table_h.nb_materials;
        ui32 k = m_materials.mat_table_h.nb_elements_total;
        ui32 mem_mat = 10*k + 80*n + 8;
        mem += mem_mat;
        print_memory("Materials", mem_mat);

        /*
        // Sources
        ui32 mem_src = 4*sources.sources.ptr_sources_dim + 4*sources.sources.data_sources_dim +
                4*sources.sources.seeds_dim + 16;
        mem += mem_src;
        print_memory("Sources", mem_src);
        */
    }

    /*
    /// Digitizer /////////////////////////////////

    // init Digitizer
    if (parameters.digitizer_flag) {
        digitizer.cpu_init_pulses(particles.stack.size);

        if (target == GPU_DEVICE) {
            digitizer.gpu_init_pulses(particles.stack.size);
        }

        // If projection acquisition
        if (digitizer.flag_projection || digitizer.flag_spect_proj) {
            digitizer.init_projection();
        }

        // Mem usage
        if (display_memory_usage_flag) {
            ui32 mem_singles = 64*digitizer.pulses.size + 4;
            mem += mem_singles;
            print_memory("Singles", mem_singles);
        }
    }
    */

    // Run time
    if (m_parameters_h.display_run_time) {
        print_time("Initialization", get_time()-t_start);
    }

    // Mem usage
    if (m_parameters_h.display_memory_usage) {
        print_memory("Total memory usage", mem);
    }
}

/*

////// :: Main functions ::

// Generate particle based on the sources (CPU version)
void SimulationBuilder::primaries_generator() {

    /// CPU ///////////////////////////////////
    if (target == CPU_DEVICE) {
//#ifdef DEBUG
        printf("CPU: primaries generator\n");
//#endif

        f64 t_start;
        if (display_run_time_flag) t_start = get_time();

        // Loop over particle slot
        ui32 id = 0;
        ui32 is = 0;
        printf("particles stack size %d \n",particles.stack.size);
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

    printf("target %d \n",target);

    /// CPU ///////////////////////////////
    if (target == CPU_DEVICE) {

#ifdef DEBUG
        printf("CPU: main navigator\n");
#endif

        f64 t_start;
        if (display_run_time_flag) t_start = get_time();

        cpu_main_navigator(particles.stack, geometry.world,
                           materials.materials_table, cs_tables.photon_CS_table, parameters,
                           digitizer.pulses, history);

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
                           digitizer.dpulses, gpu_block_size);
        
                           
        if (display_run_time_flag) {
            cudaEventRecord(t_stop);
            cudaEventSynchronize(t_stop);
            f32 time_ms = 0;
            cudaEventElapsedTime(&time_ms, t_start, t_stop);
            print_time("Main navigation", time_ms/1000.0); // in s
        }

    }

}



////// :: Getting ::

ParticleBuilder SimulationBuilder::get_particles() {
    return particles;
}

////// :: Command ::



// Start the simulation
void SimulationBuilder::start_simulation() {

    ui32 iter = 0;

    // Main loop
    printf("nb of iterations %d \n", nb_of_iterations);
    while (iter < nb_of_iterations) {

            // If history is required
            if (target == CPU_DEVICE && history.record_flag == ENABLED) history.cur_iter = iter;
            
            printf("primaries_generator \n");
            
            // Sources
            primaries_generator();
            
            // Clear gpu pulses
            if (target == GPU_DEVICE)
                digitizer.clear_gpu_pulses();
                
            digitizer.clear_cpu_pulses();
            
            printf("main_navigator \n");
            // Navigation
            main_navigator();
            
            // Process and store singles on CPU
            if (parameters.digitizer_flag) {
                f64 t_start = get_time();
                if (target == GPU_DEVICE) {
                    digitizer.copy_pulses_gpu2cpu();
                }

                // The complete chain
                digitizer.process_chain(iter, sources.tot_activity, geometry.world);

                // Run time
                if (display_run_time_flag) {
                    print_time("Process singles", get_time()-t_start);
                }
            }
            
        // iter
        ++iter;
        
        printf(">> Iter %i / %i\n", iter, nb_of_iterations);

    } // main loop
    
    // Test if one more iteration is needed
    if (nb_of_particles % particles.stack.size) {
    
        particles.stack.size = nb_of_particles - (nb_of_iterations * particles.stack.size);
        particles.dstack.size = particles.stack.size;
        
        // If history is required
        if (target == CPU_DEVICE && history.record_flag == ENABLED) history.cur_iter = iter;
        
        printf("primaries_generator \n");
        
        // Sources
        primaries_generator();
        
        // Clear gpu pulses
        if (target == GPU_DEVICE)
            digitizer.clear_gpu_pulses();
            
        digitizer.clear_cpu_pulses();
        
        printf("main_navigator \n");
        // Navigation
        main_navigator();
        
        // Process and store singles on CPU
        if (parameters.digitizer_flag) {
            f64 t_start = get_time();

            if (target == GPU_DEVICE) {
                digitizer.copy_pulses_gpu2cpu();
            }

            // The complete chain
            digitizer.process_chain(iter, sources.tot_activity, geometry.world);

            // Run time
            if (display_run_time_flag) {
                print_time("Process singles", get_time()-t_start);
            }
        }
    }
    
    // Free cpu pulses
    digitizer.free_cpu_pulses();
    
    // Free particles stack
    particles.cpu_free_stack();

    // Free materials table
    materials.free_materials_table();
    
    free(parameters.physics_list);
    free(parameters.secondaries_list); 
    
}


////// :: Utils ::





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


















*/

#endif
