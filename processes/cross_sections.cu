// GGEMS Copyright (C) 2015

/*!
 * \file cross_sections.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */


#ifndef CROSS_SECTIONS_CU
#define CROSS_SECTIONS_CU
#include "cross_sections.cuh"

//// CrossSectionsManager class ////////////////////////////////////////////////////

CrossSections::CrossSections()
{
    photon_CS.data_h.nb_bins = 0;
    photon_CS.data_h.nb_mat = 0;
    electron_CS.data_h.nb_bins = 0;
    electron_CS.data_h.nb_mat = 0;
}

//// Private - Electron ////////////////////////////////////////////////////////////

f32 CrossSections::m_get_electron_dedx(f32 energy, ui8 mat_id)
{
    ui32 E_index = binary_search ( energy, electron_CS.data_h.E, m_nb_bins );
    ui32 index = mat_id*m_nb_bins + E_index;

    f32 DeDxeIoni, DeDxeBrem;

    if ( E_index == m_nb_bins - 1 )
    {
        DeDxeIoni = electron_CS.data_h.eIonisationdedx[ index ];
        DeDxeBrem = electron_CS.data_h.eBremdedx[ index ];
    }
    else
    {
        DeDxeIoni = linear_interpolation ( electron_CS.data_h.E[index]  , electron_CS.data_h.eIonisationdedx[index],
                                           electron_CS.data_h.E[index+1], electron_CS.data_h.eIonisationdedx[index+1],
                                           energy
                                         );

        DeDxeBrem = linear_interpolation ( electron_CS.data_h.E[index]  , electron_CS.data_h.eBremdedx[index],
                                           electron_CS.data_h.E[index+1], electron_CS.data_h.eBremdedx[index+1],
                                           energy
                                         );
    }

    return DeDxeIoni + DeDxeBrem;
}

void CrossSections::m_build_electron_table()
{
    electron_CS.data_h.nb_bins = m_nb_bins;
    electron_CS.data_h.nb_mat = m_nb_mat;

    // Memory allocation for tables
    ui32 nb_tot = m_nb_mat * m_nb_bins;
    electron_CS.data_h.E = new f32[m_nb_bins];
    electron_CS.data_h.eRange = new f32[nb_tot];

    electron_CS.data_h.eIonisationCS = new f32[nb_tot];
    electron_CS.data_h.eIonisationdedx= new f32[nb_tot];

    electron_CS.data_h.eBremCS = new f32[nb_tot];
    electron_CS.data_h.eBremdedx= new f32[nb_tot];

    electron_CS.data_h.eMSC = new f32[nb_tot];

    // Fill the energy table
    f32 min_E = m_parameters.data_h.cs_table_min_E;
    f32 max_E = m_parameters.data_h.cs_table_max_E;
    f32 slope = log(max_E / min_E);    
    electron_CS.data_h.E_min = min_E;
    electron_CS.data_h.E_max = max_E;

    ui32 i = 0; while (i < m_nb_bins)
    {
        // Fill energy table with log scale
        electron_CS.data_h.E[i] = min_E * exp( slope * ( (f32)i / ( (f32)m_nb_bins-1 ) ) ) * MeV;
        ++i;
    }

    // For each material
    f32 energy;
    ui32 index;
    for ( ui32 id_mat = 0; id_mat < m_nb_mat; ++id_mat )
    {
        // For each bin
        for ( ui32 i=0; i< m_nb_bins; i++ )
        {
            // index
            index = id_mat*m_nb_bins+i;
            // energy
            energy = electron_CS.data_h.E[i];

            // Create tables if physic is activated
            if ( m_parameters.data_h.physics_list[ELECTRON_IONISATION] == true )
            {
                electron_CS.data_h.eIonisationdedx[index] = ElectronIonisation_DEDX( m_materials, energy, id_mat );
                electron_CS.data_h.eIonisationCS[index] = ElectronIonisation_CS( m_materials, energy, id_mat );
            }
            else
            {
                electron_CS.data_h.eIonisationCS[index] = 0.;
                electron_CS.data_h.eIonisationdedx[index] = 0.;
            }

            if ( m_parameters.data_h.physics_list[ELECTRON_BREMSSTRAHLUNG] == true )
            {
                electron_CS.data_h.eBremdedx[index] = ElectronBremsstrahlung_DEDX( m_materials, energy, id_mat );
                electron_CS.data_h.eBremCS[index] = ElectronBremmsstrahlung_CS ( m_materials, energy, max_E, id_mat );
            }
            else
            {
                electron_CS.data_h.eBremCS[index] = 0.;
                electron_CS.data_h.eBremdedx[index] = 0.;
            }

            if ( m_parameters.data_h.physics_list[ELECTRON_MSC] == true )
            {
                electron_CS.data_h.eMSC[index] = ElectronMultipleScattering_CS( m_materials, energy, id_mat );
            }
            else
            {
                electron_CS.data_h.eMSC[index] = 0.;
            }


        } // bins

        /// Compute the range table (after computing all dE/dx)
        index = id_mat*m_nb_bins;
        f32 eDXDE = electron_CS.data_h.eIonisationdedx[index] + electron_CS.data_h.eBremdedx[index];
        if ( eDXDE > 0.) eDXDE = 2. * electron_CS.data_h.E[0] / eDXDE;
        electron_CS.data_h.eRange[index] = eDXDE;

        // For each bin
        ui32 n = 100;
        for ( ui32 i=1; i< m_nb_bins; i++ )
        {
            f32 dE = (electron_CS.data_h.E[i] - electron_CS.data_h.E[i-1]) / n;
            energy = electron_CS.data_h.E[i] + dE*0.5;

            f32 esum = 0.0;
            ui32 j=0; while (j < n)   // 100 ? - JB
            {
                energy -= dE;
                eDXDE = m_get_electron_dedx( energy, id_mat );
                if ( eDXDE > 0.0 ) esum += ( dE / eDXDE );
                ++j;
            }

            electron_CS.data_h.eRange[index+i] = electron_CS.data_h.eRange[index+i-1] + esum;

        }



    } // mat

}

void CrossSections::m_dump_electron_tables( std::string dirname )
{

    /* NOT WORKING - JB
    std::string tmp = dirname + "E.txt";
    ImageReader::recordTables ( tmp.c_str(), 0, m_nb_bins, electron_CS.data_h.eIonisationdedx, electron_CS.data_h.eIonisationdedx );


    for ( ui32 i = 0; i< m_nb_mat; ++i )
    {
        std::string tmp = dirname + to_string ( i ) + ".txt";
        ImageReader::recordTables ( tmp.c_str(), i*m_nb_bins, ( i+1 ) *m_nb_bins,
                                    electron_CS.data_h.eIonisationdedx,
                                    electron_CS.data_h.eIonisationCS,
                                    electron_CS.data_h.eBremdedx,
                                    electron_CS.data_h.eBremCS,
                                    electron_CS.data_h.eMSC,
                                    electron_CS.data_h.eRange );
    }
    */

    ui32 i=0; while( i < m_nb_bins )
    {
        printf("E %f MeV - eIonDeDx %f eIonCS %e - eBremDeDx %f eBremCS %e\n", electron_CS.data_h.E[i],
               electron_CS.data_h.eIonisationdedx[i], electron_CS.data_h.eIonisationCS[i],
               electron_CS.data_h.eBremdedx[i], electron_CS.data_h.eBremCS[i]);
        ++i;
    }

}

/////////////////////////////////////////////////////////////////////////////


//// Private - Photon ///////////////////////////////////////////////////////

void CrossSections::m_build_photon_table()
{
    // Then init data
    ui32 tot_elt = m_nb_mat*m_nb_bins;
    //ui32 tot_elt_mem = tot_elt * sizeof(f32);
    f32 min_E = m_parameters.data_h.cs_table_min_E;
    f32 max_E = m_parameters.data_h.cs_table_max_E;

    photon_CS.data_h.Compton_Std_CS = new f32[tot_elt];
    photon_CS.data_h.Photoelectric_Std_CS = new f32[tot_elt];
    photon_CS.data_h.Photoelectric_Std_xCS = new f32[m_nb_bins*101]; // 100 Z elements,
    // starting from index 1
    photon_CS.data_h.Rayleigh_Lv_CS = new f32[tot_elt];
    photon_CS.data_h.Rayleigh_Lv_SF = new f32[m_nb_bins*101]; // 100 Z elements,
    // starting from index 1
    photon_CS.data_h.Rayleigh_Lv_xCS = new f32[m_nb_bins*101]; // 100 Z elements,
    // starting from index 1
    photon_CS.data_h.E_bins = new f32[m_nb_bins];
    photon_CS.data_h.E_min = min_E;
    photon_CS.data_h.E_max = max_E;
    photon_CS.data_h.nb_bins = m_nb_bins;
    photon_CS.data_h.nb_mat = m_nb_mat;

    // Init value
    ui32 i=0; while (i < (101*m_nb_bins)) { // 100 Z element starting from index 1
        photon_CS.data_h.Rayleigh_Lv_SF[i] = 0.0f;
        photon_CS.data_h.Rayleigh_Lv_xCS[i] = 0.0f;
        photon_CS.data_h.Photoelectric_Std_xCS[i] = 0.0f;
        ++i;
    }

    // Fill energy table with log scale
    f32 slope = log(max_E / min_E);
    i = 0;
    while (i < m_nb_bins) {
        photon_CS.data_h.E_bins[i] = min_E * exp( slope * ( (f32)i / ( (f32)m_nb_bins-1 ) ) ) * MeV;
        ++i;
    }

    // If Rayleigh scattering, load information once from G4 EM data library
    f32 *g4_ray_cs = NULL;
    f32 *g4_ray_sf = NULL;
    ui8 *flag_Z = NULL;
    if (m_parameters.data_h.physics_list[PHOTON_RAYLEIGH]) {
        g4_ray_cs = Rayleigh_CS_Livermore_load_data();
        g4_ray_sf = Rayleigh_SF_Livermore_load_data();        
    }

    if ( m_parameters.data_h.physics_list[PHOTON_RAYLEIGH] || m_parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC] )
    {
        // use to flag is scatter factor are already defined for a given Z
        flag_Z = ( ui8* )malloc( 101*sizeof( ui8 ) );
        i=0; while( i<101 ) { flag_Z[ i ] = 0; ++i; }
    }

    // Get CS for each material, energy bin and phys effect
    ui32 imat=0;
    ui32 abs_index;
    while (imat < m_nb_mat) {

        // for each energy bin
        i=0; while (i < m_nb_bins) {

            // absolute index to store data within the table
            abs_index = imat*m_nb_bins + i;

            // for each phys effect
            if (m_parameters.data_h.physics_list[PHOTON_COMPTON]) {
                photon_CS.data_h.Compton_Std_CS[abs_index] = Compton_CS_standard(m_materials, imat,
                                                                                 photon_CS.data_h.E_bins[i]);
            }
            else
            {
                photon_CS.data_h.Compton_Std_CS[abs_index] = 0.0f;
            }

            if (m_parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC]) {
                photon_CS.data_h.Photoelectric_Std_CS[abs_index] = Photoelec_CS_standard(m_materials, imat,
                                                                                         photon_CS.data_h.E_bins[i]);
            }
            else
            {
                photon_CS.data_h.Photoelectric_Std_CS[abs_index] = 0.0f;
            }

            if (m_parameters.data_h.physics_list[PHOTON_RAYLEIGH]) {
                photon_CS.data_h.Rayleigh_Lv_CS[abs_index] = Rayleigh_CS_Livermore(m_materials, g4_ray_cs,
                                                                                   imat, photon_CS.data_h.E_bins[i]);
            }
            else
            {
                photon_CS.data_h.Rayleigh_Lv_CS[abs_index] = 0.0f;
            }

            ++i;
        } // i

        // Special case for Photoelectric and Rayleigh where scatter factor and CS are needed for each Z
        if ( m_parameters.data_h.physics_list[PHOTON_RAYLEIGH] || m_parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC] ) {
            ui32 iZ, Z;
            // This table compute scatter factor for each Z (only for Z which were not already defined)
            iZ=0; while (iZ < m_materials.nb_elements[ imat ]) {

                Z = m_materials.mixture[ m_materials.index[ imat ] + iZ ];

                f32 atom_num_dens = m_materials.atom_num_dens[ m_materials.index[ imat ] + iZ ];

                // If for this Z nothing was already calculated
                if ( !flag_Z[ Z ] ) {
                    flag_Z[ Z ] = 1;

                    // for each energy bin
                    i=0; while (i < m_nb_bins) {
                        // absolute index to store data within the table
                        abs_index = Z*m_nb_bins + i;

                        if ( m_parameters.data_h.physics_list[PHOTON_RAYLEIGH] )
                        {
                            photon_CS.data_h.Rayleigh_Lv_SF[ abs_index ] = Rayleigh_SF_Livermore(g4_ray_sf,
                                                                                                 photon_CS.data_h.E_bins[i],
                                                                                                 Z);
                            photon_CS.data_h.Rayleigh_Lv_xCS[ abs_index ] = atom_num_dens *
                                                                            Rayleigh_CSPA_Livermore(g4_ray_cs, photon_CS.data_h.E_bins[i], Z);
                        }                        

                        if ( m_parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC] )
                        {                                                       
                            photon_CS.data_h.Photoelectric_Std_xCS[ abs_index ] = atom_num_dens *
                                                                                  Photoelec_CSPA_standard(photon_CS.data_h.E_bins[i], Z);
                        }

                        ++i;
                    } // i
                } // flag_Z
                ++iZ;

            } // iZ
        } // if


        ++imat;
    } // imat

    // Free mem
    free(flag_Z);

}

void CrossSections::m_dump_photon_tables( std::string dirname )
{

    std::string tmp = dirname + "E.txt";
    ImageReader::recordTables ( tmp.c_str(), 0, m_nb_bins,
                                photon_CS.data_h.E_bins, photon_CS.data_h.E_bins );

    for ( ui32 i = 0; i< m_nb_mat; ++i )
    {
        std::string tmp = dirname + to_string ( i ) + ".txt";
        ImageReader::recordTables ( tmp.c_str(), i*m_nb_bins, ( i+1 ) *m_nb_bins,
                                    photon_CS.data_h.Compton_Std_CS,
                                    photon_CS.data_h.Photoelectric_Std_CS,
                                    photon_CS.data_h.Rayleigh_Lv_CS );
    }

}

/////////////////////////////////////////////////////////////////////////////


//// Private - Misc /////////////////////////////////////////////////////////

bool CrossSections::m_check_mandatory()
{
    if (m_materials.nb_materials == 0 || m_parameters.data_h.cs_table_nbins == 0) return false;
    else return true;
}

/////////////////////////////////////////////////////////////////////////////

void CrossSections::initialize(Materials materials, GlobalSimulationParameters parameters) {

    // Store global parameters
    m_parameters = parameters;
    m_nb_bins = m_parameters.data_h.cs_table_nbins;
    m_nb_mat = materials.data_h.nb_materials;
    m_materials = materials.data_h;

    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("CrossSectionsManager parameters error!");
        exit_simulation();
    }

    // Find if there are photon and electron in this simulation;
    bool there_is_photon = m_parameters.data_h.physics_list[PHOTON_COMPTON] ||
                           m_parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC] ||
                           m_parameters.data_h.physics_list[PHOTON_RAYLEIGH];

    bool there_is_electron = m_parameters.data_h.physics_list[ELECTRON_IONISATION] ||
                             m_parameters.data_h.physics_list[ELECTRON_BREMSSTRAHLUNG] ||
                             m_parameters.data_h.physics_list[ELECTRON_MSC];

    // Build table on CPU side
    if (there_is_photon)   m_build_photon_table();
    if (there_is_electron) m_build_electron_table();

    // Allocation and copy to GPU
    if (parameters.data_h.device_target == GPU_DEVICE)
    {
        if ( there_is_photon )   m_copy_photon_cs_table_cpu2gpu();
        if ( there_is_electron ) m_copy_electron_cs_table_cpu2gpu();
    }


}

// Copy CS table to the device
void CrossSections::m_copy_photon_cs_table_cpu2gpu()
{
    ui32 n = photon_CS.data_h.nb_bins;
    ui32 k = photon_CS.data_h.nb_mat;

    // Allocate GPU mem
    HANDLE_ERROR( cudaMalloc((void**) &photon_CS.data_d.E_bins, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &photon_CS.data_d.Compton_Std_CS, n*k*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &photon_CS.data_d.Photoelectric_Std_CS, n*k*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &photon_CS.data_d.Photoelectric_Std_xCS, n*101*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &photon_CS.data_d.Rayleigh_Lv_CS, n*k*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &photon_CS.data_d.Rayleigh_Lv_SF, n*101*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &photon_CS.data_d.Rayleigh_Lv_xCS, n*101*sizeof(f32)) );

    // Copy data to GPU
    photon_CS.data_d.nb_bins = n;
    photon_CS.data_d.nb_mat = k;
    photon_CS.data_d.E_min = photon_CS.data_h.E_min;
    photon_CS.data_d.E_max = photon_CS.data_h.E_max;

    HANDLE_ERROR( cudaMemcpy(photon_CS.data_d.E_bins, photon_CS.data_h.E_bins,
                             sizeof(f32)*n, cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(photon_CS.data_d.Compton_Std_CS, photon_CS.data_h.Compton_Std_CS,
                             sizeof(f32)*n*k, cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(photon_CS.data_d.Photoelectric_Std_CS, photon_CS.data_h.Photoelectric_Std_CS,
                             sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(photon_CS.data_d.Photoelectric_Std_xCS, photon_CS.data_h.Photoelectric_Std_xCS,
                             sizeof(f32)*n*101, cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(photon_CS.data_d.Rayleigh_Lv_CS, photon_CS.data_h.Rayleigh_Lv_CS,
                             sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(photon_CS.data_d.Rayleigh_Lv_SF, photon_CS.data_h.Rayleigh_Lv_SF,
                             sizeof(f32)*n*101, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(photon_CS.data_d.Rayleigh_Lv_xCS, photon_CS.data_h.Rayleigh_Lv_xCS,
                             sizeof(f32)*n*101, cudaMemcpyHostToDevice) );
}

// Copy CS table to the device
void CrossSections::m_copy_electron_cs_table_cpu2gpu()
{
    ui32 n = electron_CS.data_h.nb_bins;
    ui32 k = electron_CS.data_h.nb_mat;

    // Allocate GPU mem
    HANDLE_ERROR( cudaMalloc((void**) &electron_CS.data_d.E              , k*n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &electron_CS.data_d.eIonisationdedx, k*n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &electron_CS.data_d.eIonisationCS  , k*n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &electron_CS.data_d.eBremdedx      , k*n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &electron_CS.data_d.eBremCS        , k*n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &electron_CS.data_d.eMSC           , k*n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &electron_CS.data_d.eRange         , k*n*sizeof(f32)) );   

    // Copy data to GPU
    electron_CS.data_d.nb_bins = n;
    electron_CS.data_d.nb_mat = k;
    electron_CS.data_d.E_min = electron_CS.data_h.E_min;
    electron_CS.data_d.E_max = electron_CS.data_h.E_max;
    electron_CS.data_d.cutEnergyElectron = electron_CS.data_h.cutEnergyElectron;
    electron_CS.data_d.cutEnergyGamma    = electron_CS.data_h.cutEnergyGamma;       

    HANDLE_ERROR( cudaMemcpy(electron_CS.data_d.E              , electron_CS.data_h.E               , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(electron_CS.data_d.eIonisationdedx, electron_CS.data_h.eIonisationdedx , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(electron_CS.data_d.eIonisationCS  , electron_CS.data_h.eIonisationCS   , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(electron_CS.data_d.eBremdedx      , electron_CS.data_h.eBremdedx       , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(electron_CS.data_d.eBremCS        , electron_CS.data_h.eBremCS         , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(electron_CS.data_d.eMSC           , electron_CS.data_h.eMSC            , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(electron_CS.data_d.eRange         , electron_CS.data_h.eRange          , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
}

#endif
