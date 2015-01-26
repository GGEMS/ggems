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

    parameters.record_dose_flag = DISABLED;
    parameters.digitizer_flag = DISABLED;
    history.record_flag = DISABLED;

}

////// :: Main functions ::

// Generate particle based on the sources (CPU version)
void SimulationBuilder::cpu_primaries_generator() {

    // Loop over particle slot
    ui32 id = 0;
    ui32 is = 0;
    while (id < particles.stack.size) {

        // TODO - Generic and multi-sources
        //      Read CDF sources
        //      Rnd sources
        is = 0; // first source

        // Read the address source
        ui32 adr = sources.sources.ptr_sources[is];

        // Read the kind of sources
        ui32 type = (ui32)(sources.sources.data_sources[adr+ADR_SRC_TYPE]);
        ui32 geom_id = (ui32)(sources.sources.data_sources[adr+ADR_SRC_GEOM_ID]);

        // Point Source
        if (type == POINT_SOURCE) {
            f32 px = sources.sources.data_sources[adr+ADR_POINT_SRC_PX];
            f32 py = sources.sources.data_sources[adr+ADR_POINT_SRC_PY];
            f32 pz = sources.sources.data_sources[adr+ADR_POINT_SRC_PZ];
            f32 energy = sources.sources.data_sources[adr+ADR_POINT_SRC_ENERGY];

            point_source_primary_generator(particles.stack, id, px, py, pz, energy, PHOTON, geom_id);

        } else if (type == CONE_BEAM_SOURCE) {
            f32 px = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_PX];
            f32 py = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_PY];
            f32 pz = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_PZ];
            f32 phi = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_PHI];
            f32 theta = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_THETA];
            f32 psi = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_PSI];
            f32 aperture = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_APERTURE];
            f32 energy = sources.sources.data_sources[adr+ADR_CONE_BEAM_SRC_ENERGY];

            cone_beam_source_primary_generator(particles.stack, id, px, py, pz,
                                               phi, theta, psi, aperture, energy, PHOTON, geom_id);
        } else if (type == VOXELIZED_SOURCE) {

            f32 px = sources.sources.data_sources[adr+ADR_VOX_SOURCE_PX];
            f32 py = sources.sources.data_sources[adr+ADR_VOX_SOURCE_PY];
            f32 pz = sources.sources.data_sources[adr+ADR_VOX_SOURCE_PZ];

            f32 nb_vox_x = sources.sources.data_sources[adr+ADR_VOX_SOURCE_NB_VOX_X];
            f32 nb_vox_y = sources.sources.data_sources[adr+ADR_VOX_SOURCE_NB_VOX_Y];
            f32 nb_vox_z = sources.sources.data_sources[adr+ADR_VOX_SOURCE_NB_VOX_Z];

            f32 sx = sources.sources.data_sources[adr+ADR_VOX_SOURCE_SPACING_X];
            f32 sy = sources.sources.data_sources[adr+ADR_VOX_SOURCE_SPACING_Y];
            f32 sz = sources.sources.data_sources[adr+ADR_VOX_SOURCE_SPACING_Z];

            f32 energy = sources.sources.data_sources[adr+ADR_VOX_SOURCE_ENERGY];

            f32 nb_acts = sources.sources.data_sources[adr+ADR_VOX_SOURCE_NB_CDF];

            f32 emission_type = sources.sources.data_sources[adr+ADR_VOX_SOURCE_EMISSION_TYPE];

            f32 *cdf_index = &(sources.sources.data_sources[adr+ADR_VOX_SOURCE_CDF_INDEX]);
            ui32 adr_cdf_act = adr+nb_acts;
            f32 *cdf_act = &(sources.sources.data_sources[adr_cdf_act+ADR_VOX_SOURCE_CDF_INDEX]);

            if (emission_type == EMISSION_BACK2BACK) {
                voxelized_source_primary_generator(particles.stack, id,
                                                   cdf_index, cdf_act, nb_acts,
                                                   px, py, pz, nb_vox_x, nb_vox_y, nb_vox_z,
                                                   sx, sy, sz, energy, PHOTON, geom_id);
                // Back2back fills the particle' stack with two particles, we need to
                // adjust the ID to be the ID of event (half size) and not the ID of particles
                // Consequently ID is incremented to consider the additional particle in the stack
                ++id;

            } else if (emission_type == EMISSION_MONO) {
                printf("ERROR: voxelized source, emission 'MONO' is not impleted yet!\n");
                exit(EXIT_FAILURE);
            }

        }

        // Next particle
        ++id;

    } // i

    // History record (use only for VRML view)
    if (history.record_flag == ENABLED) {
        id=0; while (id < particles.stack.size) {
            // Record the first position for the tracking history
            history.cpu_new_particle_track(PHOTON);
            history.cpu_record_a_step(particles.stack, id);
            ++id;
        }
    }

}

// Main navigation on CPU
void SimulationBuilder::cpu_main_navigation() {

    cpu_main_navigator(particles.stack, geometry.world,
                       materials.materials_table, cs_tables.photon_CS_table, parameters,
                       digitizer.singles, history);

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

////// :: Getting ::

ParticleBuilder SimulationBuilder::get_particles() {
    return particles;
}

////// :: Command ::


// Init simualtion
void SimulationBuilder::init_simulation() {

    // First compute the number of iterations and the size of a stack // TODO Can be improved - JB
    if (nb_of_particles % particles.stack.size) {
        nb_of_iterations = (nb_of_particles / particles.stack.size) + 1;
    } else {
        nb_of_iterations = nb_of_particles / particles.stack.size;
    }
    particles.stack.size = nb_of_particles / nb_of_iterations;
    nb_of_particles = particles.stack.size * nb_of_iterations;


//    // Reset and set GPU ID and compute grid size
//    wrap_reset_device();
//    wrap_set_device(m_gpu_id);
//    m_grid_size = (m_stack_size + m_block_size - 1) / m_block_size;

//    // copy data to the device
//    wrap_copy_phantom_to_device(h_phantom, d_phantom);
//    wrap_copy_materials_to_device(h_materials, d_materials);

//    // init particle stack
//    wrap_init_particle_stack(d_particles, m_stack_size);

//    // init particle seeds
//    wrap_init_particle_seeds(d_particles, m_seed);

//    // copy the physics list to the device
//    wrap_copy_physics_list_to_device(m_physics_list);

//    // copy the secondaries list to the device
//    wrap_copy_secondaries_list_to_device(m_secondaries_list);


    if (target == CPU_DEVICE) {

        // Init the particle stack
        particles.cpu_malloc_stack();
        particles.init_stack_seed();

    }

    // Init Cross sections and physics table
    cs_tables.build_table(materials.materials_table, parameters);
    //cs_tables.print();

    // init Digitizer
    if (parameters.digitizer_flag) {
        digitizer.init_singles(particles.stack.size);
    }
}

// Start the simulation
void SimulationBuilder::start_simulation() {

    ui32 iter = 0;

    if (target == CPU_DEVICE) {

        // Main loop
        while (iter < nb_of_iterations) {
            // If history is required
            if (history.record_flag == ENABLED) history.cur_iter = iter;

            // Sources
            cpu_primaries_generator();

            // Locate the first particle position within the geometry

            // Navigation
            cpu_main_navigation();

            // Process and store singles
            if (parameters.digitizer_flag) {
                digitizer.process_singles(iter);
                digitizer.export_singles();
            }

            // iter
            ++iter;
            printf(">> Iter %i / %i\n", iter, nb_of_iterations);
        } // main loop

    }

}

////// :: Utils ::

#endif
