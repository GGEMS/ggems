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

#ifndef GGEMS_CUH
#define GGEMS_CUH

#include "global.cuh"
#include "ggems_vsource.cuh"
#include "particles.cuh"
#include "cross_sections.cuh"
#include "materials.cuh"
#include "source_manager.cuh"
#include "phantom_manager.cuh"
#include "detector_manager.cuh"

class GGEMS {
    public:
        GGEMS();
        ~GGEMS();


        // Set simulation object
//        void set_geometry(GeometryBuilder obj);
//        void set_materials(MaterialBuilder tab);
//        void set_sources(SourceBuilder src);
//        void set_particles(ParticleBuilder p);
//        void set_digitizer(Digitizer dig);

        // Setting parameters
        void set_hardware_target(std::string value);
        void set_GPU_ID(ui32 valid);
        void set_GPU_block_size(ui32 val);
        void set_process(std::string process_name);
        void set_secondary(std::string pname);
        void set_particle_cut(std::string pname, f32 E);
        void set_number_of_particles(ui64 nb);
        void set_size_of_particles_batch(ui64 nb);
        void set_CS_table_nbins(ui32 valbin);
        void set_CS_table_E_min(f32 valE);
        void set_CS_table_E_max(f32 valE);
        void set_seed(ui32 vseed);
        // Setting simulation objects
        void set_source(PointSource &aSource);
        void set_phantom(VoxPhanImgNav &aPhantom);
        // TODO DETECTOR

        // Utils
        void set_display_run_time();
        void set_display_memory_usage();

        // Main functions
        void init_simulation();
        void start_simulation();



    private:
        // Particles handler
        ParticleManager m_particles_manager;

        // Cross section handler
        CrossSectionsManager m_cross_sections;

        // Materials handler
        MaterialManager m_materials;

        // Source manager
        SourcesManager m_sources;

        // Phantom manager
        PhantomManager m_phantoms;

        // TODO Detector manager
        DetectorManager m_detectors;

        // Main parameters
        bool m_check_mandatory();
        void m_copy_parameters_cpu2gpu();
        GlobalSimulationParameters m_parameters;

        /*
        // Main functions
        void primaries_generator();
        void main_navigator();

        // For GPU





        ui32 seed;
        */

};



#endif
