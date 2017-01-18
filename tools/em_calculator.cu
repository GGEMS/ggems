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

void EmCalculator::compute_photon_tracking_uncorrelated_model( std::string mat_name, ui32 nb_samples,
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
    pFile = fopen ( "gTrackModel.raw", "wb" );

    // Some vars
    f32xyz dd = {1.0f, 0.0f, 0.0f};    // default direction
    ui32 nb_bins_lut = 10000;          // for LCDF

    f32 *cdf_dist = new f32[ nb_energy_bins*nb_bins ];
    f32 *cdf_scatter = new f32[ nb_energy_bins*nb_bins ];
    f32 *cdf_edep = new f32[ nb_energy_bins*nb_bins ];

    // For LCDF (LUT-CDF)
    ui16 *lcdf_dist = new ui16[ nb_energy_bins*nb_bins_lut ];
    ui16 *lcdf_scatter = new ui16[ nb_energy_bins*nb_bins_lut ];
    ui16 *lcdf_edep = new ui16[ nb_energy_bins*nb_bins_lut ];

    f32 *values_dist = new f32[ nb_bins ];
    f32 *values_scatter = new f32[ nb_bins ];
    f32 *values_edep = new f32[ nb_bins ];
    f32 *values_energy = new f32[ nb_energy_bins ];

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

        i++;
    }

    // Compute CDF spacing
    f32 low_dist = 2.0 *mm;
    ui32 half_nb_bins = nb_bins / 2;
    f32 di_lowdist = ( low_dist ) / f32( half_nb_bins );  // remove (-1) to end with a value = (low_dist - di_lowdist)
    f32 di_highdist = ( max_dist - low_dist ) / f32( half_nb_bins - 1 );

    f32 di_scatter = ( pi ) / f32( nb_bins - 1 );
    f32 di_edep = ( max_edep ) / f32( nb_bins - 1 );
    f32 di_energy = ( max_energy - min_energy ) / f32( nb_energy_bins - 1);

    // for LCDF
    f32 di_lut = 1.0 / (nb_bins_lut - 1);

    // Some vars for histogramm
    ui32 posi; f32 dist; ui8 process; f32 edep; f32 angle;

    // Compute bin values
    i = 0; while ( i < nb_bins )
    {
        values_energy[ i ] = min_energy + i*di_energy;

        if ( i < half_nb_bins ) values_dist[ i ] = i*di_lowdist;
        else                    values_dist[ i ] = low_dist + (i - half_nb_bins)*di_highdist;
        values_edep[ i ] = i*di_edep;
        values_scatter[ i ] = i*di_scatter;

        //printf("bin %i val %f\n", i, values_dist[ i ]);

        i++;
    }

    // Build CDF model for each energy bin value
    ui32 index; f32 energy; ui16 index_cdf;
    //ui32 ct_PE = 0;   // Count the nb of photoelectric effect
    for ( ui32 ie = 0; ie < nb_energy_bins; ie++)
    {
        energy = values_energy[ ie ];
        index  = ie*nb_bins;

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
            if ( dist < low_dist )
            {
                posi = dist / di_lowdist;
            }
            else
            {
                posi = ( dist - low_dist ) / di_highdist;
                posi += half_nb_bins;
            }

            // Not within the dist max
            if ( posi >= nb_bins )
            {
                continue;
            }

            cdf_dist[ index+posi ]++;

            // If PE, record proba
            process = m_part_manager.particles.data_h.next_discrete_process[0];

            // Get scattering and the dropped energy
            photon_resolve_discrete_process( m_part_manager.particles.data_h, m_params.data_h, m_cross_sections.photon_CS.data_h,
                                             m_materials.tables.data_h, mat_id, 0 );

            if ( process == PHOTON_PHOTOELECTRIC )
            {
                //ct_PE++;
                //continue;
                m_part_manager.particles.data_h.E[0] = 0.0;
            }

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
        //proba_PE[ ie ] = f32(ct_PE) / f32(nb_samples);

        // Build the LCDF (LUT-CDF) for Dist
        f32 p_rnd;
        index_cdf = 0;
        i = 0; while ( i < nb_bins_lut )
        {
            p_rnd = i*di_lut;
            while ( cdf_dist[ index+index_cdf ] < p_rnd && index_cdf < nb_bins )
            {
                index_cdf++;
            }
            lcdf_dist[ i + ie*nb_bins_lut ] = index_cdf;

            //printf("i %i  p_rnd %f  index_cdf %i   cdf_dist %f\n", i, p_rnd, index_cdf, cdf_dist[ index+index_cdf ]);

            i++;
        }

        // for edep
        index_cdf = 0;
        i = 0; while ( i < nb_bins_lut )
        {
            p_rnd = i*di_lut;
            while ( cdf_edep[ index+index_cdf ] < p_rnd && index_cdf < nb_bins )
            {
                index_cdf++;
            }
            lcdf_edep[ i + ie*nb_bins_lut ] = index_cdf;
            i++;
        }

        // for scatter
        index_cdf = 0;
        i = 0; while ( i < nb_bins_lut )
        {
            p_rnd = i*di_lut;
            while ( cdf_scatter[ index+index_cdf ] < p_rnd && index_cdf < nb_bins )
            {
                index_cdf++;
            }
            lcdf_scatter[ i + ie*nb_bins_lut ] = index_cdf;
            i++;
        }


    } // energy bin loop

    // Export data:
    //   Format for one material
    //     [... values_energy ...]   (nb_energy_bins)
    //     [... values_dist ...]     (nb_bins)
    //     [... values_edep ...]     (nb_bins)
    //     [... values_scatter ...]  (nb_bins)
    //////////////////////////////////////////////////////////     [... proba_PE ...]        (nb_energy_bins)
    //     [... cdf_dist ...]        (nb_bins*nb_energy_bins)
    //     [... cdf_edep ...]        (nb_bins*nb_energy_bins)
    //     [... cdf_scatter ...]     (nb_bins*nb_energy_bins)
    // OR
    //     [... lcdf_dist ...]        (nb_bins_lut*nb_energy_bins)
    //     [... lcdf_edep ...]        (nb_bins_lut*nb_energy_bins)
    //     [... lcdf_scatter ...]     (nb_bins_lut*nb_energy_bins)

    fwrite( values_energy, sizeof( f32 ), nb_energy_bins, pFile );
    fwrite( values_dist, sizeof( f32 ), nb_bins, pFile );
    fwrite( values_edep, sizeof( f32 ), nb_bins, pFile );
    fwrite( values_scatter, sizeof( f32 ), nb_bins, pFile );
    //fwrite( proba_PE, sizeof( f32 ), nb_energy_bins, pFile );

/*
    fwrite( cdf_dist, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
    fwrite( cdf_edep, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
    fwrite( cdf_scatter, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
*/
    fwrite( lcdf_dist, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );
    fwrite( lcdf_edep, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );
    fwrite( lcdf_scatter, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );

    fclose( pFile );

    pFile = fopen ( "im_lutdist.raw", "wb" );
    fwrite( lcdf_dist, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );
    fclose( pFile );


    pFile = fopen ( "im_dist.raw", "wb" );
    fwrite( cdf_dist, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
    fclose( pFile );

/*
    pFile = fopen ( "im_edep.raw", "wb" );
    fwrite( cdf_edep, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
    fclose( pFile );

    pFile = fopen ( "im_scatter.raw", "wb" );
    fwrite( cdf_scatter, sizeof( f32 ), nb_bins*nb_energy_bins, pFile );
    fclose( pFile );
*/

}


//// CORRELATED MODEL
//
// Per material:
//  1.  Lambda(E)
//  2.  Theta(Lambda)
//  3.  dE(Theta)
void EmCalculator::compute_photon_tracking_correlated_model( std::string mat_name, ui32 nb_samples,
                                                             f32 min_energy, f32 max_energy, ui32 nb_energy_bins,
                                                             f32 max_step, ui32 nb_step_bins,
                                                             ui32 nb_theta_bins,
                                                             f32 max_edep, ui32 nb_edep_bins )
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
    pFile = fopen ( "gTrackModel.raw", "wb" );

    // Some vars
    f32xyz dd = {1.0f, 0.0f, 0.0f};    // default direction
//    ui32 nb_bins_lut = 10000;          // for LCDF

    f32 *cdf_dist = new f32[ nb_energy_bins*nb_step_bins ];      // Lambda(E)
    f32 *cdf_scatter = new f32[ nb_step_bins*nb_theta_bins ];    // Theta(lambda)
    f32 *cdf_edep = new f32[ nb_theta_bins*nb_edep_bins ];      // dE(theta)
    f32 *proba_overstep = new f32[ nb_energy_bins ];
    f32 *proba_PE = new f32[ nb_energy_bins ];

    // For LCDF (LUT-CDF)
//    ui16 *lcdf_dist = new ui16[ nb_energy_bins*nb_bins_lut ];
//    ui16 *lcdf_scatter = new ui16[ nb_energy_bins*nb_bins_lut ];
//    ui16 *lcdf_edep = new ui16[ nb_energy_bins*nb_bins_lut ];

    // Init values
    ui32 i = 0; while( i < nb_energy_bins*nb_step_bins )
    {
        cdf_dist[ i++ ] = 0.0;
    }
    i = 0; while( i < nb_step_bins*nb_theta_bins )
    {
        cdf_scatter[ i++ ] = 0.0;
    }
    i = 0; while( i < nb_theta_bins*nb_edep_bins )
    {
        cdf_edep[ i++ ] = 0.0;
    }
    i = 0; while( i < nb_energy_bins )
    {
        proba_overstep[ i ] = 0.0;
        proba_PE[ i++ ] = 0.0;
    }

    // Compute CDF spacing
    f32 di_dist = ( max_step ) / f32( nb_step_bins - 1 );
    f32 di_scatter = ( pi ) / f32( nb_theta_bins - 1 );
    f32 di_edep = ( max_edep ) / f32( nb_edep_bins - 1 );
    f32 di_energy = ( max_energy - min_energy ) / f32( nb_energy_bins - 1);

    // for LCDF
//    f32 di_lut = 1.0 / (nb_bins_lut - 1);

    // Some vars for histogramm
    ui32 pos_step; ui32 pos_theta; ui32 pos_edep; f32 dist; ui8 process; f32 edep; f32 angle;
/*
    // Compute bin values
    i = 0; while ( i < nb_bins )
    {
        values_energy[ i ] = min_energy + i*di_energy;

        if ( i < half_nb_bins ) values_dist[ i ] = i*di_lowdist;
        else                    values_dist[ i ] = max_substep + (i - half_nb_bins)*di_highdist;
        values_edep[ i ] = i*di_edep;
        values_scatter[ i ] = i*di_scatter;

        //printf("bin %i val %f\n", i, values_dist[ i ]);

        i++;
    }
*/
    // Build CDF model
    f32 energy; //ui16 index_cdf;
    ui32 pos_energy;
    ui32 ct_oversteps; ui32 ct_PE;
    ui32 ct_steps; ui32 ct_process;
    for ( pos_energy = 0; pos_energy < nb_energy_bins; pos_energy++)
    {
        energy = min_energy + pos_energy*di_energy;
        ct_oversteps = 0;
        ct_steps = 0;
        ct_process = 0;
        ct_PE = 0;
        printf("Energy %f MeV\n", energy);

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
            ct_steps++;

            // To get the probabilty of the over stepping
            if (dist > max_step+di_dist)
            {
                ct_oversteps++;
                continue;
            }

            // Get process
            ct_process++;
            process = m_part_manager.particles.data_h.next_discrete_process[0];
            if ( process == PHOTON_PHOTOELECTRIC )
            {
                ct_PE++;
                continue;
            }

            // Get scattering and the dropped energy
            photon_resolve_discrete_process( m_part_manager.particles.data_h, m_params.data_h, m_cross_sections.photon_CS.data_h,
                                             m_materials.tables.data_h, mat_id, 0 );

            // edep
            edep = energy - m_part_manager.particles.data_h.E[0];

            // scatter
            angle = acosf( fxyz_dot( make_f32xyz( m_part_manager.particles.data_h.dx[0],
                                                  m_part_manager.particles.data_h.dy[0],
                                                  m_part_manager.particles.data_h.dz[0] ), dd ) );

            // Get position within each vector
            pos_theta = angle / di_scatter;
            pos_edep = edep / di_edep;
            pos_step = dist / di_dist;

//            if (pos_step == (nb_step_bins-1))
//            {
//                printf("pos_step = %i   angle = %f\n", pos_step, angle);
//            }

#ifdef DEBUG
            assert( pos_theta < nb_theta_bins );
            assert( pos_edep < nb_edep_bins );
            assert( pos_step < nb_step_bins );
#endif
            // Assign values
            cdf_dist[ pos_energy*nb_step_bins + pos_step ]++;
            cdf_scatter[ pos_step*nb_theta_bins + pos_theta ]++;
            cdf_edep[ pos_theta*nb_edep_bins + pos_edep ]++;

        } // Samples loop

        // Compute the probability to over-stepping
        proba_overstep[ pos_energy ] = f32( ct_oversteps ) / f32( ct_steps );
        proba_PE[ pos_energy ] = f32( ct_PE ) / f32( ct_process );

/*
        // Build the LCDF (LUT-CDF) for Dist
        f32 p_rnd;
        index_cdf = 0;
        i = 0; while ( i < nb_bins_lut )
        {
            p_rnd = i*di_lut;
            while ( cdf_dist[ index+index_cdf ] < p_rnd && index_cdf < nb_bins )
            {
                index_cdf++;
            }
            lcdf_dist[ i + ie*nb_bins_lut ] = index_cdf;

            //printf("i %i  p_rnd %f  index_cdf %i   cdf_dist %f\n", i, p_rnd, index_cdf, cdf_dist[ index+index_cdf ]);

            i++;
        }

        // for edep
        index_cdf = 0;
        i = 0; while ( i < nb_bins_lut )
        {
            p_rnd = i*di_lut;
            while ( cdf_edep[ index+index_cdf ] < p_rnd && index_cdf < nb_bins )
            {
                index_cdf++;
            }
            lcdf_edep[ i + ie*nb_bins_lut ] = index_cdf;
            i++;
        }

        // for scatter
        index_cdf = 0;
        i = 0; while ( i < nb_bins_lut )
        {
            p_rnd = i*di_lut;
            while ( cdf_scatter[ index+index_cdf ] < p_rnd && index_cdf < nb_bins )
            {
                index_cdf++;
            }
            lcdf_scatter[ i + ie*nb_bins_lut ] = index_cdf;
            i++;
        }
*/

    } // energy bin loop

    //// Compute CDF for each row of each matrices

    f64 sum;

    // lambda(E)
    pos_energy = 0; while ( pos_energy < nb_energy_bins )
    {
        // sum
        sum = 0.0;
        ui32 index = pos_energy*nb_step_bins;
        pos_step = index; while ( pos_step < (index + nb_step_bins) )
        {
            sum += cdf_dist[ pos_step ];
            pos_step++;
        }
        if (sum == 0)
        {
            printf("Warning: sum 0 lambda(E)\n");
        }
        // norm
        pos_step = index; while ( pos_step < (index + nb_step_bins) )
        {
            cdf_dist[ pos_step ] /= sum;
            pos_step++;
        }
        // CDF
        pos_step = index+1; while ( pos_step < (index + nb_step_bins) )
        {
            cdf_dist[ pos_step ] += cdf_dist[ pos_step-1 ];
            pos_step++;
        }

        pos_energy++;
    }

    // theta(lambda)
    pos_step = 0; while ( pos_step < nb_step_bins )
    {
        // sum
        sum = 0.0;
        ui32 index = pos_step*nb_theta_bins;
        pos_theta = index; while ( pos_theta < (index + nb_theta_bins) )
        {
            sum += cdf_scatter[ pos_theta ];
            pos_theta++;
        }
        if (sum == 0)
        {
            printf("Warning: sum 0 theta(lambda)\n");
        }
        // norm
        pos_theta = index; while ( pos_theta < (index + nb_theta_bins) )
        {
            cdf_scatter[ pos_theta ] /= sum;
            pos_theta++;
        }
        // CDF
        pos_theta = index+1; while ( pos_theta < (index + nb_theta_bins) )
        {
            cdf_scatter[ pos_theta ] += cdf_scatter[ pos_theta-1 ];
            pos_theta++;
        }

        pos_step++;
    }

    // edep(theta)
    pos_theta = 0; while ( pos_theta < nb_theta_bins )
    {
        // sum
        sum = 0.0;
        ui32 index = pos_theta*nb_edep_bins;
        pos_edep = index; while ( pos_edep < (index + nb_edep_bins) )
        {
            sum += cdf_edep[ pos_edep ];
            pos_edep++;
        }
        if (sum == 0)
        {
            printf("Warning: sum 0 edep(theta)\n");
        }
        // norm
        pos_edep = index; while ( pos_edep < (index + nb_edep_bins) )
        {
            cdf_edep[ pos_edep ] /= sum;
            pos_edep++;
        }
        // CDF
        pos_edep = index+1; while ( pos_edep < (index + nb_edep_bins) )
        {
            cdf_edep[ pos_edep ] += cdf_edep[ pos_edep-1 ];
            pos_edep++;
        }

        pos_theta++;
    }

    //fwrite( &max_step, sizeof( f32 ), 1, pFile );
    fwrite( &di_energy, sizeof( f32 ), 1, pFile );
    fwrite( &di_dist, sizeof( f32 ), 1, pFile );
    fwrite( &di_scatter, sizeof( f32 ), 1, pFile );
    fwrite( &di_edep, sizeof( f32 ), 1, pFile );

    fwrite( cdf_dist, sizeof( f32 ), nb_step_bins*nb_energy_bins, pFile );
    fwrite( cdf_scatter, sizeof( f32 ), nb_step_bins*nb_theta_bins, pFile );
    fwrite( cdf_edep, sizeof( f32 ), nb_theta_bins*nb_edep_bins, pFile );
    fwrite( proba_overstep, sizeof( f32 ), nb_energy_bins, pFile );

/*
    fwrite( lcdf_dist, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );
    fwrite( lcdf_edep, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );
    fwrite( lcdf_scatter, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );
*/

    fclose( pFile );

//    pFile = fopen ( "im_lutdist.raw", "wb" );
//    fwrite( lcdf_dist, sizeof( ui16 ), nb_bins_lut*nb_energy_bins, pFile );
//    fclose( pFile );


    pFile = fopen ( "im_dist.raw", "wb" );
    fwrite( cdf_dist, sizeof( f32 ), nb_step_bins*nb_energy_bins, pFile );
    fclose( pFile );

    pFile = fopen ( "im_edep.raw", "wb" );
    fwrite( cdf_edep, sizeof( f32 ), nb_theta_bins*nb_edep_bins, pFile );
    fclose( pFile );

    pFile = fopen ( "im_scatter.raw", "wb" );
    fwrite( cdf_scatter, sizeof( f32 ), nb_step_bins*nb_theta_bins, pFile );
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
