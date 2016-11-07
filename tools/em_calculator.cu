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

void EmCalculator::compute_photon_cdf_track( std::string mat_name, ui32 nb_samples,
                                             f32 min_energy, f32 max_energy, ui32 nb_energy_bins,
                                             f32 max_dist, f32 max_edep, ui32 nb_bins )
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

    // Open the output file
    FILE *pFile;
    pFile = fopen ( "mixmodel.raw", "wb" );

    // Some vars
    f32xyz dd = {1.0f, 0.0f, 0.0f};    // default direction

    f32 *cdf_dist = new f32[ nb_energy_bins*nb_bins ];
    f32 *cdf_scatter = new f32[ nb_energy_bins*nb_bins ];
    f32 *cdf_edep = new f32[ nb_energy_bins*nb_bins ];

    f32 *values_dist = new f32[ nb_bins ];
    f32 *values_scatter = new f32[ nb_bins ];
    f32 *values_edep = new f32[ nb_bins ];
    f32 *values_energy = new f32[ nb_energy_bins ];

    f32 *proba_PE = new f32[ nb_energy_bins ];

    // Init values
    ui32 i = 0; while( i < nb_energy_bins*nb_bins )
    {
        cdf_dist[ i ] = 0.0;
        cdf_scatter[ i ] = 0.0;
        cdf_edep[ i ] = 0.0;

        i++;
    }
    i = 0; while( i < nb_bins )
    {
        values_dist[ i ] = 0.0;
        values_scatter[ i ] = 0.0;
        values_edep[ i ] = 0.0;

        proba_PE[ i ] = 0.0;

        i++;
    }

    // Compute CDF spacing
    f32 di_dist = ( max_dist ) / f32( nb_bins - 1 );
    f32 di_scatter = ( pi ) / f32( nb_bins - 1 );
    f32 di_edep = ( max_edep ) / f32( nb_bins - 1 );
    f32 di_energy = ( max_energy - min_energy ) / f32( nb_energy_bins - 1);

    // Some vars for histogramm
    ui32 posi; f32 dist; ui8 process; f32 edep; f32 angle;

    // Compute bin values
    i = 0; while ( i < nb_bins )
    {
        values_energy[ i ] = min_energy + i*di_energy;
        values_dist[ i ] = i*di_dist;
        values_edep[ i ] = i*di_edep;
        values_scatter[ i ] = i*di_scatter;

        i++;
    }

    // Build CDF model for each energy bin value
    ui32 index;
    ui32 ct_PE = 0;   // Count the nb of photoelectric effect
    for ( ui32 ie = 0; ie < nb_energy_bins; ie++)
    {
        energy = values_energy[ ie ];
        index  = ie*nb_bins;
        ct_PE  = 0.0;

        for ( ui32 is = 0; is < nb_samples; is++ )
        {
            // Init a particle
            m_part_manager.particles.data_h.E[0] = energy;                             // Energy in MeV

            m_part_manager.particles.data_h.px[0] = 0.0f;                              // Position in mm
            m_part_manager.particles.data_h.py[0] = 0.0f;                              //
            m_part_manager.particles.data_h.pz[0] = 0.0f;                              //

            m_part_manager.particles.data_h.dx[0] = dd.x;                              // Direction (unit vector)
            m_part_manager.particles.data_h.dy[0] = dd.y;                              //
            m_part_manager.particles.data_h.dz[0] = dd.z;                              //

            // Get step distance
            photon_get_next_interaction( m_part_manager.particles.data_h, m_params.data_h, m_cross_sections.photon_CS.data_h, mat_id, 0 );
            dist = m_part_manager.particles.data_h.next_interaction_distance[0];

            //particles.E_index[part_id] = E_index;
            posi = dist / di_dist;
#ifdef DEBUG
            assert( posi < nb_bins );
#endif
            cdf_dist[ index+posi ]++;

            // If PE, record proba
            process = m_part_manager.particles.data_h.next_discrete_process[0];
            if ( process == PHOTON_PHOTOELECTRIC )
            {
                ct_PE++;
                continue;
            }

            // Get scattering and the dropped energy
            photon_resolve_discrete_process( m_part_manager.particles.data_h, m_params.data_h, m_cross_sections.photon_CS.data_h,
                                             m_materials.data_h, mat_id, 0 );

            // edep
            edep = energy - m_part_manager.particles.data_h.E[0];
            posi = edep / di_edep;
#ifdef DEBUG
            assert( posi < nb_bins );
#endif
            cdf_edep[ index+posi ]++;

            // scatter
            angle = acosf( fxyz_dot( make_f32xyz( m_part_manager.particles.data_h.dx[0],
                                                  m_part_manager.particles.data_h.dy[0],
                                                  m_part_manager.particles.data_h.dz[0] ), dd ) );
            posi = angle / di_scatter;
#ifdef DEBUG
            assert( posi < nb_bins );
#endif
            cdf_scatter[ index+posi ]++;

        } // Samples loop

        // Get the sum CDF values
        f64 sum_dist = 0;
        f64 sum_edep = 0;
        f64 sum_scatter = 0;
        i = index; while( i < (index+nb_bins) )
        {
            sum_dist += cdf_dist[ i ];
            sum_edep += cdf_edep[ i ];
            sum_scatter += cdf_scatter[ i ];
            i++;
        }

        // Normalize CDF (TODO this loop can be combine this the next one)
        i = index; while( i < (index+nb_bins) )
        {
            cdf_dist[ i ] = cdf_dist[ i ] / sum_dist;
            cdf_edep[ i ] = cdf_edep[ i ] / sum_edep;
            cdf_scatter[ i ] = cdf_scatter[ i ] / sum_scatter;
            i++;
        }

        // Compute the final CDF (cummulative)
        i = index+1; while (i < (index+nb_bins) )
        {
            cdf_dist[ i ] += cdf_dist[ i-1 ];
            cdf_edep[ i ] += cdf_edep[ i-1 ];
            cdf_scatter[ i ] += cdf_scatter[ i-1 ];
            i++;
        }

        // Compute the final PE effect proba
        proba_PE[ ie ] = f32(ct_PE) / f32(nb_samples);

    } // energy bin loop

    // Export data:
    //   Format for one material
    //     [... values_energy ...]   (nb_energy_bins)
    //     [... values_dist ...]     (nb_bins)
    //     [... values_edep ...]     (nb_bins)
    //     [... values_scatter ...]  (nb_bins)
    //     [... proba_PE ...]        (nb_energy_bins)
    //     [... cdf_dist ...]        (nb_bins*nb_energy_bins)
    //     [... cdf_edep ...]        (nb_bins*nb_energy_bins)
    //     [... cdf_scatter ...]     (nb_bins*nb_energy_bins)

    fwrite( values_energy, sizeof( f32 ), nb_energy_bins, pFile );
    fwrite( values_dist, sizeof( f32 ), nb_bins, pFile );
    fwrite( values_edep, sizeof( f32 ), nb_bins, pFile );
    fwrite( values_scatter, sizeof( f32 ), nb_bins, pFile );
    fwrite( proba_PE, sizeof( f32 ), nb_energy_bins, pFile );
    fwrite( cdf_dist, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
    fwrite( cdf_edep, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
    fwrite( cdf_scatter, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );

    fclose( pFile );
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
