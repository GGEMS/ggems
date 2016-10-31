// GGEMS Copyright (C) 2015

/*!
 * \file em_calculator.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 31/10/2016
 *
 *
 */

#ifndef EM_CALCULATOR_CU
#define EM_CALCULATOR_CU

#include "em_calculator.cuh"

/// Class //////////////////////////////////////////////

EmCalculator::EmCalculator()
{
    // Default simulation parameters
    m_params.data_h.physics_list = ( bool* ) malloc ( NB_PROCESSES*sizeof ( bool ) );
    m_params.data_h.secondaries_list = ( bool* ) malloc ( NB_PARTICLES*sizeof ( bool ) );

    ui32 i = 0;
    while ( i < NB_PROCESSES )
    {
        m_params.data_h.physics_list[i] = ENABLED;
        ++i;
    }
    i = 0;
    while ( i < NB_PARTICLES )
    {
        m_params.data_h.secondaries_list[i] = DISABLED;
        ++i;
    }

    // Parameters
    m_params.data_h.nb_of_particles = 1;
    m_params.data_h.size_of_particles_batch = 1;
    m_params.data_h.nb_of_batches = 1;
    m_params.data_h.time = 0;
    m_params.data_h.seed = 123456789;
    m_params.data_h.cs_table_nbins = 220;
    m_params.data_h.cs_table_min_E = 990*eV;
    m_params.data_h.cs_table_max_E = 250*MeV;
    m_params.data_h.photon_cut = 1 *um;
    m_params.data_h.electron_cut = 1 *um;
    m_params.data_h.nb_of_secondaries = 0;
    m_params.data_h.geom_tolerance = 100.0 *nm;
    m_params.data_h.device_target = CPU_DEVICE;
    m_params.data_h.gpu_id = 0;
    m_params.data_h.gpu_block_size = 192;
    m_params.data_h.display_run_time = ENABLED;
    m_params.data_h.display_memory_usage = DISABLED;
    m_params.data_h.display_energy_cuts = DISABLED;
    m_params.data_h.verbose = ENABLED;

    m_mat_names_db.clear();

}

EmCalculator::~EmCalculator()
{

}

void EmCalculator::initialize(std::string materials_db_filename)
{
    // Load materials
    m_materials.load_materials_database( materials_db_filename );

    // Get list of all materials name from the database
    m_mat_names_db = m_get_all_materials_name( materials_db_filename );

    // Init materials with all names from the database
    m_materials.initialize( m_mat_names_db, m_params );

    // Compute all cross sections
    m_cross_sections.initialize( m_materials, m_params );

    // Init particle stack
    m_part_manager.initialize( m_params );
}

void EmCalculator::compute_photon_cdf_track(std::string mat_name, f32 energy, i32 n)
{
    // Get mat index
    i32 mat_id=0; while( mat_id < m_mat_names_db.size() )
    {
        if ( m_mat_names_db[mat_id] == mat_name ) break;
        ++mat_id;
    }

    // Set up the particle
    m_part_manager.particles.data_h.tof[0] = 0.0f;                             // Time of flight
    m_part_manager.particles.data_h.endsimu[0] = PARTICLE_ALIVE;               // Status of the particle

    m_part_manager.particles.data_h.level[0] = PRIMARY;                        // It is a primary particle
    m_part_manager.particles.data_h.pname[0] = PHOTON;                          // a photon or an electron

    m_part_manager.particles.data_h.geometry_id[0] = 0;                        // Some internal variables
    m_part_manager.particles.data_h.next_discrete_process[0] = NO_PROCESS;     //
    m_part_manager.particles.data_h.next_interaction_distance[0] = 0.0;        //


    ui32 i=0; while (i < n)
    {
        m_part_manager.particles.data_h.E[0] = energy;                             // Energy in MeV

        m_part_manager.particles.data_h.px[0] = 0.0f;                              // Position in mm
        m_part_manager.particles.data_h.py[0] = 0.0f;                              //
        m_part_manager.particles.data_h.pz[0] = 0.0f;                              //

        m_part_manager.particles.data_h.dx[0] = 1.0f;                              // Direction (unit vector)
        m_part_manager.particles.data_h.dy[0] = 0.0f;                              //
        m_part_manager.particles.data_h.dz[0] = 0.0f;                              //

        // Get step distance
        photon_get_next_interaction( m_part_manager.particles.data_h, m_params.data_h, m_cross_sections.photon_CS.data_h, mat_id, 0 );

        // print results
        f32 dist = m_part_manager.particles.data_h.next_interaction_distance[0];
        ui8 process = m_part_manager.particles.data_h.next_discrete_process[0];
        //particles.E_index[part_id] = E_index;

        printf("i %i  Dist %f  proc %i ", i, dist, process);

        // If PE, record proba

        // Get scattering and energy droped
        photon_resolve_discrete_process( m_part_manager.particles.data_h, m_params.data_h, m_cross_sections.photon_CS.data_h,
                                         m_materials.data_h, mat_id, 0 );

        printf("E %f  dir %f %f %f\n", m_part_manager.particles.data_h.E[0],
                                       m_part_manager.particles.data_h.dx[0],
                                       m_part_manager.particles.data_h.dy[0],
                                       m_part_manager.particles.data_h.dz[0]);

        i++;
    }
}


/// Private ///////////////////////////////////////////

std::vector< std::string > EmCalculator::m_get_all_materials_name(std::string filename)
{
    TxtReader *txt_reader = new TxtReader;
    std::ifstream file(filename.c_str());
    std::string line;

    std::vector< std::string > names;

    while (file)
    {
        txt_reader->skip_comment(file);
        std::getline(file, line);

        if (file)
        {
            std::string aName = txt_reader->read_material_name(line);
            //printf("name %s\n", aName.c_str());
            names.push_back( aName );

            // Skip elements definition
            i16 nbelts = txt_reader->read_material_nb_elements(line);
            ui16 i=0; while (i < nbelts) {
                std::getline(file, line);
                ++i;
            }

        }

    }

    delete txt_reader;

    return names;
}





#endif
