// GGEMS Copyright (C) 2017

/*!
 * \file cross_sections.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 18 novembre 2015
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 */


#ifndef CROSS_SECTIONS_CU
#define CROSS_SECTIONS_CU
#include "cross_sections.cuh"

//// CrossSectionsManager class ////////////////////////////////////////////////////

CrossSections::CrossSections()
{
    h_photon_CS = nullptr;
    d_photon_CS = nullptr;

    h_electron_CS = nullptr;
    d_electron_CS = nullptr;

    m_nb_bins = 0;
    m_nb_mat = 0;
}

//// Private - Electron ////////////////////////////////////////////////////////////

f32 CrossSections::m_get_electron_dedx(f32 energy, ui8 mat_id)
{
    ui32 E_index = binary_search ( energy, h_electron_CS->E, m_nb_bins );
    ui32 index = mat_id*m_nb_bins + E_index;

    f32 DeDxeIoni, DeDxeBrem;

    if ( E_index == m_nb_bins - 1 )
    {
        DeDxeIoni = h_electron_CS->eIonisationdedx[ index ];
        DeDxeBrem = h_electron_CS->eBremdedx[ index ];
    }
    else
    {
        DeDxeIoni = linear_interpolation ( h_electron_CS->E[index]  , h_electron_CS->eIonisationdedx[index],
                                           h_electron_CS->E[index+1], h_electron_CS->eIonisationdedx[index+1],
                                           energy
                                         );

        DeDxeBrem = linear_interpolation ( h_electron_CS->E[index]  , h_electron_CS->eBremdedx[index],
                                           h_electron_CS->E[index+1], h_electron_CS->eBremdedx[index+1],
                                           energy
                                         );
    }

    return DeDxeIoni + DeDxeBrem;
}

void CrossSections::m_build_electron_table()
{    

    // Struct allocation
    ui32 nb_tot = m_nb_mat * m_nb_bins;
    h_electron_CS = (ElectronsCrossSectionData*)malloc( sizeof(ElectronsCrossSectionData) );

    h_electron_CS->E = new f32[m_nb_bins];
    h_electron_CS->eRange = new f32[nb_tot];

    h_electron_CS->eIonisationCS = new f32[nb_tot];
    h_electron_CS->eIonisationdedx= new f32[nb_tot];

    h_electron_CS->eBremCS = new f32[nb_tot];
    h_electron_CS->eBremdedx= new f32[nb_tot];

    h_electron_CS->eMSC = new f32[nb_tot];

    h_electron_CS->eIonisation_E_CS_max = new f32[m_nb_mat];
    h_electron_CS->eIonisation_CS_max = new f32[m_nb_mat];

    h_electron_CS->nb_bins = m_nb_bins;
    h_electron_CS->nb_mat = m_nb_mat;

    // Fill the energy table
    f32 min_E = mh_parameters->cs_table_min_E;
    f32 max_E = mh_parameters->cs_table_max_E;
    f32 slope = log(max_E / min_E);    
    h_electron_CS->E_min = min_E;
    h_electron_CS->E_max = max_E;

    ui32 i = 0; while (i < m_nb_bins)
    {
        // Fill energy table with log scale
        h_electron_CS->E[i] = min_E * exp( slope * ( (f32)i / ( (f32)m_nb_bins-1 ) ) ) * MeV;
        ++i;
    }

    // For each material
    f32 energy;
    ui32 index;
    for ( ui32 id_mat = 0; id_mat < m_nb_mat; ++id_mat )
    {
        // Prepare vars
        h_electron_CS->eIonisation_E_CS_max[ id_mat ] = 0;
        h_electron_CS->eIonisation_CS_max[ id_mat ] = 0;

        // For each bin
        for ( ui32 i=0; i< m_nb_bins; i++ )
        {
            // index
            index = id_mat*m_nb_bins+i;
            // energy
            energy = h_electron_CS->E[i];

            // Create tables if physic is activated
            if ( mh_parameters->physics_list[ELECTRON_IONISATION] == true )
            {
                h_electron_CS->eIonisationdedx[index] = ElectronIonisation_DEDX( mh_materials, energy, id_mat );
                h_electron_CS->eIonisationCS[index] = ElectronIonisation_CS( mh_materials, energy, id_mat );

                // Search for the max CS value
                if ( h_electron_CS->eIonisationCS[ index ] >  h_electron_CS->eIonisation_CS_max[ id_mat ] )
                {
                    h_electron_CS->eIonisation_CS_max[ id_mat ] = h_electron_CS->eIonisationCS[ index ];
                    h_electron_CS->eIonisation_E_CS_max[ id_mat ] = energy;
                }

            }
            else
            {
                h_electron_CS->eIonisationCS[index] = 0.;
                h_electron_CS->eIonisationdedx[index] = 0.;
            }

            if ( mh_parameters->physics_list[ELECTRON_BREMSSTRAHLUNG] == true )
            {
                h_electron_CS->eBremdedx[index] = ElectronBremsstrahlung_DEDX( mh_materials, energy, id_mat );
                h_electron_CS->eBremCS[index] = ElectronBremmsstrahlung_CS ( mh_materials, energy, max_E, id_mat );


            }
            else
            {
                h_electron_CS->eBremCS[index] = 0.;
                h_electron_CS->eBremdedx[index] = 0.;
            }

            if ( mh_parameters->physics_list[ELECTRON_MSC] == true )
            {
                h_electron_CS->eMSC[index] = ElectronMultipleScattering_CS( mh_materials, energy, id_mat );
            }
            else
            {
                h_electron_CS->eMSC[index] = 0.;
            }


        } // bins

        /// Compute the range table (after computing all dE/dx)
        index = id_mat*m_nb_bins;
        f32 eDXDE = h_electron_CS->eIonisationdedx[index] + h_electron_CS->eBremdedx[index];
        if ( eDXDE > 0. ) eDXDE = 2. * h_electron_CS->E[0] / eDXDE;
        h_electron_CS->eRange[index] = eDXDE;

        // For each bin
        ui32 n = 100;
        for ( ui32 i=1; i < m_nb_bins; i++ )
        {
            f32 dE = (h_electron_CS->E[i] - h_electron_CS->E[i-1]) / n;
            energy = h_electron_CS->E[i] + dE*0.5;

            f32 esum = 0.0;
            ui32 j=0; while (j < n)   // 100 ? - JB
            {
                energy -= dE;
                eDXDE = m_get_electron_dedx( energy, id_mat );
                if ( eDXDE > 0.0 ) esum += ( dE / eDXDE );
                ++j;
            }

            h_electron_CS->eRange[index+i] = h_electron_CS->eRange[index+i-1] + esum;

        }

    } // mat

}

void CrossSections::m_dump_electron_tables( std::string dirname )
{


    ui32 i=0; while( i < m_nb_bins )
    {
        printf("E %f MeV - eIonDeDx %f eIonCS %e - eBremDeDx %f eBremCS %e\n", h_electron_CS->E[i],
               h_electron_CS->eIonisationdedx[i], h_electron_CS->eIonisationCS[i],
               h_electron_CS->eBremdedx[i], h_electron_CS->eBremCS[i]);
        ++i;
    }

}

/////////////////////////////////////////////////////////////////////////////


//// Private - Photon ///////////////////////////////////////////////////////

void CrossSections::m_build_photon_table()
{
    // Then init data
    ui32 tot_elt = m_nb_mat*m_nb_bins;    
    f32 min_E = mh_parameters->cs_table_min_E;
    f32 max_E = mh_parameters->cs_table_max_E;

    // Struct allocation
    h_photon_CS = (PhotonCrossSectionData*)malloc( sizeof(PhotonCrossSectionData) );

    h_photon_CS->Compton_Std_CS = new f32[tot_elt];
    h_photon_CS->Photoelectric_Std_CS = new f32[tot_elt];
    h_photon_CS->Photoelectric_Std_xCS = new f32[m_nb_bins*101]; // 100 Z elements,
    // starting from index 1
    h_photon_CS->Rayleigh_Lv_CS = new f32[tot_elt];
    h_photon_CS->Rayleigh_Lv_SF = new f32[m_nb_bins*101]; // 100 Z elements,
    // starting from index 1
    h_photon_CS->Rayleigh_Lv_xCS = new f32[m_nb_bins*101]; // 100 Z elements,
    // starting from index 1
    h_photon_CS->E_bins = new f32[m_nb_bins];
    h_photon_CS->E_min = min_E;
    h_photon_CS->E_max = max_E;
    h_photon_CS->nb_bins = m_nb_bins;
    h_photon_CS->nb_mat = m_nb_mat;

    // Init value
    ui32 i=0; while (i < (101*m_nb_bins)) { // 100 Z element starting from index 1
        h_photon_CS->Rayleigh_Lv_SF[i] = 0.0f;
        h_photon_CS->Rayleigh_Lv_xCS[i] = 0.0f;
        h_photon_CS->Photoelectric_Std_xCS[i] = 0.0f;
        ++i;
    }

    // Fill energy table with log scale
    f32 slope = log(max_E / min_E);
    i = 0;
    while (i < m_nb_bins) {
        h_photon_CS->E_bins[i] = min_E * exp( slope * ( (f32)i / ( (f32)m_nb_bins-1 ) ) ) * MeV;
        ++i;
    }

    // If Rayleigh scattering, load information once from G4 EM data library
    f32 *g4_ray_cs = NULL;
    f32 *g4_ray_sf = NULL;
    ui8 *flag_Z = NULL;
    if (mh_parameters->physics_list[PHOTON_RAYLEIGH]) {
        g4_ray_cs = Rayleigh_CS_Livermore_load_data();
        g4_ray_sf = Rayleigh_SF_Livermore_load_data();        
    }

    if ( mh_parameters->physics_list[PHOTON_RAYLEIGH] || mh_parameters->physics_list[PHOTON_PHOTOELECTRIC] )
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
            if (mh_parameters->physics_list[PHOTON_COMPTON]) {
                h_photon_CS->Compton_Std_CS[abs_index] = Compton_CS_standard( mh_materials, imat,
                                                                               h_photon_CS->E_bins[i] );
            }
            else
            {
                h_photon_CS->Compton_Std_CS[abs_index] = 0.0f;
            }

            if (mh_parameters->physics_list[PHOTON_PHOTOELECTRIC]) {
                h_photon_CS->Photoelectric_Std_CS[abs_index] = Photoelec_CS_standard( mh_materials, imat,
                                                                                        h_photon_CS->E_bins[i] );
            }
            else
            {
                h_photon_CS->Photoelectric_Std_CS[abs_index] = 0.0f;
            }

            if (mh_parameters->physics_list[PHOTON_RAYLEIGH]) {
                h_photon_CS->Rayleigh_Lv_CS[abs_index] = Rayleigh_CS_Livermore( mh_materials, g4_ray_cs,
                                                                                 imat, h_photon_CS->E_bins[i] );
            }
            else
            {
                h_photon_CS->Rayleigh_Lv_CS[abs_index] = 0.0f;
            }

            ++i;
        } // i

        // Special case for Photoelectric and Rayleigh where scatter factor and CS are needed for each Z
        if ( mh_parameters->physics_list[PHOTON_RAYLEIGH] || mh_parameters->physics_list[PHOTON_PHOTOELECTRIC] ) {
            ui32 iZ, Z;
            // This table compute scatter factor for each Z (only for Z which were not already defined)
            iZ=0; while (iZ < mh_materials->nb_elements[ imat ]) {

                Z = mh_materials->mixture[ mh_materials->index[ imat ] + iZ ];

                f32 atom_num_dens = mh_materials->atom_num_dens[ mh_materials->index[ imat ] + iZ ];

                // If for this Z nothing was already calculated
                if ( !flag_Z[ Z ] ) {
                    flag_Z[ Z ] = 1;

                    // for each energy bin
                    i=0; while (i < m_nb_bins) {
                        // absolute index to store data within the table
                        abs_index = Z*m_nb_bins + i;

                        if ( mh_parameters->physics_list[PHOTON_RAYLEIGH] )
                        {
                            h_photon_CS->Rayleigh_Lv_SF[ abs_index ] = Rayleigh_SF_Livermore( g4_ray_sf,
                                                                                               h_photon_CS->E_bins[i],
                                                                                               Z );
                            h_photon_CS->Rayleigh_Lv_xCS[ abs_index ] = atom_num_dens *
                                                                            Rayleigh_CSPA_Livermore(g4_ray_cs, h_photon_CS->E_bins[i], Z);
                        }                        

                        if ( mh_parameters->physics_list[PHOTON_PHOTOELECTRIC] )
                        {                                                       
                            h_photon_CS->Photoelectric_Std_xCS[ abs_index ] = atom_num_dens *
                                                                                  Photoelec_CSPA_standard(h_photon_CS->E_bins[i], Z);
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



/////////////////////////////////////////////////////////////////////////////


//// Private - Misc /////////////////////////////////////////////////////////

bool CrossSections::m_check_mandatory()
{
    if (mh_materials->nb_materials == 0 || mh_parameters->cs_table_nbins == 0) return false;
    else return true;
}

/////////////////////////////////////////////////////////////////////////////

void CrossSections::initialize(const MaterialsData *h_materials, const GlobalSimulationParametersData *h_parameters) {

    // Store global parameters
    mh_parameters = h_parameters;
    mh_materials = h_materials;

    m_nb_bins = h_parameters->cs_table_nbins;
    m_nb_mat = h_materials->nb_materials;


    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("CrossSectionsManager parameters error!");
        exit_simulation();
    }

    // Find if there are photon and electron in this simulation;
    bool there_is_photon = h_parameters->physics_list[PHOTON_COMPTON] ||
                           h_parameters->physics_list[PHOTON_PHOTOELECTRIC] ||
                           h_parameters->physics_list[PHOTON_RAYLEIGH];

    bool there_is_electron = h_parameters->physics_list[ELECTRON_IONISATION] ||
                             h_parameters->physics_list[ELECTRON_BREMSSTRAHLUNG] ||
                             h_parameters->physics_list[ELECTRON_MSC];

    // Build table on CPU side
    if (there_is_photon)   m_build_photon_table();
    if (there_is_electron) m_build_electron_table();

    // Allocation and copy to GPU    
    if (there_is_photon) m_copy_photon_cs_table_cpu2gpu();
    if (there_is_electron) m_copy_electron_cs_table_cpu2gpu();

}

// Copy CS table to the device
void CrossSections::m_copy_photon_cs_table_cpu2gpu()
{
    ui32 n = h_photon_CS->nb_bins;
    ui32 k = h_photon_CS->nb_mat;

    /// First, struct allocation
    HANDLE_ERROR( cudaMalloc( (void**) &d_photon_CS, sizeof( PhotonCrossSectionData ) ) );

    /// Device pointers allocation
    f32 *E_bins;  // n
    HANDLE_ERROR( cudaMalloc((void**) &E_bins, n*sizeof(f32)) );

    f32* Compton_Std_CS;        // n*k
    HANDLE_ERROR( cudaMalloc((void**) &Compton_Std_CS, n*k*sizeof(f32)) );

    f32* Photoelectric_Std_CS;  // n*k
    HANDLE_ERROR( cudaMalloc((void**) &Photoelectric_Std_CS, n*k*sizeof(f32)) );
    f32* Photoelectric_Std_xCS; // n*101 (Nb of Z)
    HANDLE_ERROR( cudaMalloc((void**) &Photoelectric_Std_xCS, n*101*sizeof(f32)) );

    f32* Rayleigh_Lv_CS;        // n*k
    HANDLE_ERROR( cudaMalloc((void**) &Rayleigh_Lv_CS, n*k*sizeof(f32)) );
    f32* Rayleigh_Lv_SF;        // n*101 (Nb of Z)
    HANDLE_ERROR( cudaMalloc((void**) &Rayleigh_Lv_SF, n*101*sizeof(f32)) );
    f32* Rayleigh_Lv_xCS;       // n*101 (Nb of Z)
    HANDLE_ERROR( cudaMalloc((void**) &Rayleigh_Lv_xCS, n*101*sizeof(f32)) );

    /// Copy host data to device
    HANDLE_ERROR( cudaMemcpy( E_bins, h_photon_CS->E_bins,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( Compton_Std_CS, h_photon_CS->Compton_Std_CS,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( Photoelectric_Std_CS, h_photon_CS->Photoelectric_Std_CS,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( Photoelectric_Std_xCS, h_photon_CS->Photoelectric_Std_xCS,
                              n*101*sizeof(f32), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( Rayleigh_Lv_CS, h_photon_CS->Rayleigh_Lv_CS,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( Rayleigh_Lv_SF, h_photon_CS->Rayleigh_Lv_SF,
                              n*101*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( Rayleigh_Lv_xCS, h_photon_CS->Rayleigh_Lv_xCS,
                              n*101*sizeof(f32), cudaMemcpyHostToDevice ) );


    /// Bind data to the struct
    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->E_bins), &E_bins,
                              sizeof(d_photon_CS->E_bins), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->Compton_Std_CS), &Compton_Std_CS,
                              sizeof(d_photon_CS->Compton_Std_CS), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->Photoelectric_Std_CS), &Photoelectric_Std_CS,
                              sizeof(d_photon_CS->Photoelectric_Std_CS), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->Photoelectric_Std_xCS), &Photoelectric_Std_xCS,
                              sizeof(d_photon_CS->Photoelectric_Std_xCS), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->Rayleigh_Lv_CS), &Rayleigh_Lv_CS,
                              sizeof(d_photon_CS->Rayleigh_Lv_CS), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->Rayleigh_Lv_SF), &Rayleigh_Lv_SF,
                              sizeof(d_photon_CS->Rayleigh_Lv_SF), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->Rayleigh_Lv_xCS), &Rayleigh_Lv_xCS,
                              sizeof(d_photon_CS->Rayleigh_Lv_xCS), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->E_min), &(h_photon_CS->E_min),
                              sizeof(d_photon_CS->E_min), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->E_max), &(h_photon_CS->E_max),
                              sizeof(d_photon_CS->E_max), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->nb_bins), &n,
                              sizeof(d_photon_CS->nb_bins), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_photon_CS->nb_mat), &k,
                              sizeof(d_photon_CS->nb_mat), cudaMemcpyHostToDevice ) );


}

// Copy CS table to the device
void CrossSections::m_copy_electron_cs_table_cpu2gpu()
{
    ui32 n = h_electron_CS->nb_bins;
    ui32 k = h_electron_CS->nb_mat;

    /// First, struct allocation
    HANDLE_ERROR( cudaMalloc( (void**) &d_electron_CS, sizeof( ElectronsCrossSectionData ) ) );

    /// Device pointers allocation
    f32* E;                    // n*k
    HANDLE_ERROR( cudaMalloc((void**) &E, n*k*sizeof(f32)) );
    f32* eIonisationdedx;      // n*k
    HANDLE_ERROR( cudaMalloc((void**) &eIonisationdedx, n*k*sizeof(f32)) );
    f32* eIonisationCS;        // n*k
    HANDLE_ERROR( cudaMalloc((void**) &eIonisationCS, n*k*sizeof(f32)) );
    f32* eBremdedx;            // n*k
    HANDLE_ERROR( cudaMalloc((void**) &eBremdedx, n*k*sizeof(f32)) );
    f32* eBremCS;              // n*k
    HANDLE_ERROR( cudaMalloc((void**) &eBremCS, n*k*sizeof(f32)) );
    f32* eMSC;                 // n*k
    HANDLE_ERROR( cudaMalloc((void**) &eMSC, n*k*sizeof(f32)) );
    f32* eRange;               // n*k
    HANDLE_ERROR( cudaMalloc((void**) &eRange, n*k*sizeof(f32)) );
    f32* eIonisation_E_CS_max; // k  |_ For CS from eIoni
    HANDLE_ERROR( cudaMalloc((void**) &eIonisation_E_CS_max, k*sizeof(f32)) );
    f32* eIonisation_CS_max;   // k  |
    HANDLE_ERROR( cudaMalloc((void**) &eIonisation_CS_max, k*sizeof(f32)) );

    /// Copy host data to device
    HANDLE_ERROR( cudaMemcpy( E, h_electron_CS->E,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eIonisationdedx, h_electron_CS->eIonisationdedx,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eIonisationCS, h_electron_CS->eIonisationCS,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eBremdedx, h_electron_CS->eBremdedx,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eBremCS, h_electron_CS->eBremCS,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eMSC, h_electron_CS->eMSC,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eRange, h_electron_CS->eRange,
                              n*k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eIonisation_E_CS_max, h_electron_CS->eIonisation_E_CS_max,
                              k*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( eIonisation_CS_max, h_electron_CS->eIonisation_CS_max,
                              k*sizeof(f32), cudaMemcpyHostToDevice ) );

    /// Bind data to the struct
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->E), &E,
                              sizeof(d_electron_CS->E), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eIonisationdedx), &eIonisationdedx,
                              sizeof(d_electron_CS->eIonisationdedx), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eIonisationCS), &eIonisationCS,
                              sizeof(d_electron_CS->eIonisationCS), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eBremdedx), &eBremdedx,
                              sizeof(d_electron_CS->eBremdedx), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eBremCS), &eBremCS,
                              sizeof(d_electron_CS->eBremCS), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eMSC), &eMSC,
                              sizeof(d_electron_CS->eMSC), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eRange), &eRange,
                              sizeof(d_electron_CS->eRange), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eIonisation_E_CS_max), &eIonisation_E_CS_max,
                              sizeof(d_electron_CS->eIonisation_E_CS_max), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->eIonisation_CS_max), &eIonisation_CS_max,
                              sizeof(d_electron_CS->eIonisation_CS_max), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->E_min), &(h_electron_CS->E_min),
                              sizeof(d_electron_CS->E_min), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->E_max), &(h_electron_CS->E_max),
                              sizeof(d_electron_CS->E_max), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->nb_bins), &(h_electron_CS->nb_bins),
                              sizeof(d_electron_CS->nb_bins), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->nb_mat), &(h_electron_CS->nb_mat),
                              sizeof(d_electron_CS->nb_mat), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->cutEnergyElectron), &(h_electron_CS->cutEnergyElectron),
                              sizeof(d_electron_CS->cutEnergyElectron), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_electron_CS->cutEnergyGamma), &(h_electron_CS->cutEnergyGamma),
                              sizeof(d_electron_CS->cutEnergyGamma), cudaMemcpyHostToDevice ) );

}














#endif
