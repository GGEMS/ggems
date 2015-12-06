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

CrossSectionsManager::CrossSectionsManager() {
    photon_CS.nb_bins = 0;
    photon_CS.nb_mat = 0;
}

// Main function
bool CrossSectionsManager::m_check_mandatory() {
    if (photon_CS.nb_bins == 0 || photon_CS.nb_mat == 0) return false;
    else return true;
}

void CrossSectionsManager::initialize(Materials materials, GlobalSimulationParameters parameters) {

    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("CrossSectionsManager paramters error!");
        exit_simulation();
    }

    // Build table on CPU side
    m_build_table(materials, parameters);

    // Allocation and copy
    if (parameters.data_h.device_target == GPU_DEVICE) m_copy_cs_table_cpu2gpu();

}

// Build cross sections table according material, physics effects and particles
void CrossSectionsManager::m_build_table(Materials materials, GlobalSimulationParameters parameters) {

    // Read parameters
    ui32 nbin = parameters.data_h.cs_table_nbins;
    f32 min_E = parameters.data_h.cs_table_min_E;
    f32 max_E = parameters.data_h.cs_table_max_E;

    // First thing first, sample energy following the number of bins
    f32 slope = log(max_E / min_E);
    ui32 i = 0;
    photon_CS.data_h.E_bins = (f32*)malloc(nbin * sizeof(f32));
    while (i < nbin) {
        photon_CS.data_h.E_bins[i] = min_E * exp( slope * ((f32)i/((f32)nbin-1)) ) * MeV;
        ++i;
    }

    // Find if there are photon and electron in this simulation;
    i8 there_is_photon = parameters.data_h.physics_list[PHOTON_COMPTON] ||
                         parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC] ||
                         parameters.data_h.physics_list[PHOTON_RAYLEIGH];
    //i8 there_is_electron = parameters.physics_list[ELECTRON_IONISATION] ||
    //                         parameters.physics_list[ELECTRON_BREMSSTRAHLUNG] ||
    //                         parameters.physics_list[ELECTRON_MSC];

    // Then init data
    ui32 tot_elt = materials.nb_materials*nbin;
    ui32 tot_elt_mem = tot_elt * sizeof(f32);

    // Photon CS table if need
    if (there_is_photon) {
        photon_CS.data_h.Compton_Std_CS = (f32*)malloc(tot_elt_mem);
        photon_CS.data_h.Photoelectric_Std_CS = (f32*)malloc(tot_elt_mem);
        photon_CS.data_h.Photoelectric_Std_xCS = (f32*)malloc(nbin * 101 * sizeof(f32)); // 100 Z elements,
                                                                                            // starting from index 1
        photon_CS.data_h.Rayleigh_Lv_CS = (f32*)malloc(tot_elt_mem);
        photon_CS.data_h.Rayleigh_Lv_SF = (f32*)malloc(nbin * 101 * sizeof(f32)); // 100 Z elements,
                                                                                     // starting from index 1
        photon_CS.data_h.Rayleigh_Lv_xCS = (f32*)malloc(nbin * 101 * sizeof(f32)); // 100 Z elements,
                                                                                     // starting from index 1       
        photon_CS.data_h.E_min = min_E;
        photon_CS.data_h.E_max = max_E;
        photon_CS.data_h.nb_bins = nbin;
        photon_CS.data_h.nb_mat = materials.nb_materials;
        photon_CS.nb_bins = nbin;
        photon_CS.nb_mat = materials.nb_materials;

        // Init value
        i=0; while (i < tot_elt) {            
            photon_CS.data_h.Compton_Std_CS[i] = 0.0f;
            photon_CS.data_h.Photoelectric_Std_CS[i] = 0.0f;
            photon_CS.data_h.Rayleigh_Lv_CS[i] = 0.0f;
            ++i;
        }
        i=0; while (i < (101*nbin)) { // 100 Z element starting from index 1
            photon_CS.data_h.Rayleigh_Lv_SF[i] = 0.0f;
            photon_CS.data_h.Rayleigh_Lv_xCS[i] = 0.0f;
            photon_CS.data_h.Photoelectric_Std_xCS[i] = 0.0f;
            ++i;
        }

    }
    
    // idem for e- table - TODO
    // if (there_is_electron) {
    // ...
    // ...

    // If Rayleigh scattering, load information once from G4 EM data library
    f32 *g4_ray_cs = NULL;
    f32 *g4_ray_sf = NULL;
    i8 *flag_Z = NULL;
    if (parameters.data_h.physics_list[PHOTON_RAYLEIGH]) {

        g4_ray_cs = Rayleigh_CS_Livermore_load_data();
        g4_ray_sf = Rayleigh_SF_Livermore_load_data();

        // use to flag is scatter factor are already defined for a given Z
        flag_Z = (i8*)malloc(101*sizeof(i8));
        i=0; while(i<101) {flag_Z[i]=0; ++i;}
    }
    
    // Get CS for each material, energy bin and phys effect
    ui32 imat=0;
    ui32 abs_index;
    while (imat < materials.nb_materials) {

        // for each energy bin
        i=0; while (i < nbin) {

            // absolute index to store data within the table
            abs_index = imat*nbin + i;

            // for each phys effect
            if (parameters.data_h.physics_list[PHOTON_COMPTON]) {
                photon_CS.data_h.Compton_Std_CS[abs_index] = Compton_CS_standard(materials.data_h, imat,
                                                                                 photon_CS.data_h.E_bins[i]);
            }
            if (parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC]) {
                photon_CS.data_h.Photoelectric_Std_CS[abs_index] = Photoelec_CS_standard(materials.data_h, imat,
                                                                                         photon_CS.data_h.E_bins[i]);
            }
            if (parameters.data_h.physics_list[PHOTON_RAYLEIGH]) {
                photon_CS.data_h.Rayleigh_Lv_CS[abs_index] = Rayleigh_CS_Livermore(materials.data_h, g4_ray_cs,
                                                                                   imat, photon_CS.data_h.E_bins[i]);
            }

            // TODO
            // idem with Electron_CS_table

            ++i;
        } // i              

        // Special case for Photoelectric and Rayleigh where scatter factor and CS are needed for each Z
        if (parameters.data_h.physics_list[PHOTON_RAYLEIGH]) {
            ui32 iZ, Z;
            // This table compute scatter factor for each Z (only for Z which were not already defined)
            iZ=0; while (iZ < materials.data_h.nb_elements[imat]) {
                Z = materials.data_h.mixture[materials.data_h.index[imat]+iZ];

                f32 atom_num_dens = materials.data_h.atom_num_dens[materials.data_h.index[imat]+iZ];

                // If for this Z nothing was already calculated
                if (!flag_Z[Z]) {
                    flag_Z[Z] = 1;

                    // for each energy bin
                    i=0; while (i < nbin) {
                        // absolute index to store data within the table
                        abs_index = Z*nbin + i;
                        photon_CS.data_h.Rayleigh_Lv_SF[abs_index] = Rayleigh_SF_Livermore(g4_ray_sf,
                                                                                           photon_CS.data_h.E_bins[i],
                                                                                           Z);

                        photon_CS.data_h.Rayleigh_Lv_xCS[abs_index] = atom_num_dens *
                                                                      Rayleigh_CSPA_Livermore(g4_ray_cs, photon_CS.data_h.E_bins[i], Z);

                        photon_CS.data_h.Photoelectric_Std_xCS[abs_index] = atom_num_dens *
                                                                            Photoelec_CSPA_standard(photon_CS.data_h.E_bins[i], Z);

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

/*
// Print CS talbe (for debugging)
void CrossSectionsManager::print() {

    ui32 imat, iE, abs_index;


    printf("::::::::::::::::::::::::::::::::::::::::::::\n");
    printf("::::::::::::::::: Gamma ::::::::::::::::::::\n");
    printf("::::::::::::::::::::::::::::::::::::::::::::\n\n");

    printf("==== Compton Standard CS ====\n\n");

    imat=0; while (imat < photon_CS_table_h.nb_mat) {
        printf("## Material %i\n", imat);
        iE=0; while (iE < photon_CS_table_h.nb_bins) {
            abs_index = imat*photon_CS_table_h.nb_bins + iE;
            printf("E %e CS %e\n", photon_CS_table_h.E_bins[iE],
                                   photon_CS_table_h.Compton_Std_CS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");

    printf("==== Photoelectric Standard CS ====\n");

    imat=0; while (imat < photon_CS_table_h.nb_mat) {
        printf("## Material %i\n", imat);
        iE=0; while (iE < photon_CS_table_h.nb_bins) {
            abs_index = imat*photon_CS_table_h.nb_bins + iE;
            printf("E %e CS %e\n", photon_CS_table_h.E_bins[iE],
                                   photon_CS_table_h.Photoelectric_Std_CS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");

    imat=0; while (imat < 101) {
        printf("## Z %i\n", imat);
        iE=0; while (iE < photon_CS_table_h.nb_bins) {
            abs_index = imat*photon_CS_table_h.nb_bins + iE;
            printf("E %e CS %e\n", photon_CS_table_h.E_bins[iE],
                                   photon_CS_table_h.Photoelectric_Std_xCS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");

    printf("==== Rayleigh Livermore CS ====\n");

    imat=0; while (imat < photon_CS_table_h.nb_mat) {
        printf("## Material %i\n", imat);
        iE=0; while (iE < photon_CS_table_h.nb_bins) {
            abs_index = imat*photon_CS_table_h.nb_bins + iE;
            printf("E %e CS %e\n", photon_CS_table_h.E_bins[iE],
                                   photon_CS_table_h.Rayleigh_Lv_CS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");

    printf("==== Rayleigh Livermore SF ====\n");

    imat=0; while (imat < 101) {
        printf("## Z %i\n", imat);
        iE=0; while (iE < photon_CS_table_h.nb_bins) {
            abs_index = imat*photon_CS_table_h.nb_bins + iE;
            printf("E %e SF %e CS %e\n", photon_CS_table_h.E_bins[iE],
                                         photon_CS_table_h.Rayleigh_Lv_SF[abs_index],
                                         photon_CS_table_h.Rayleigh_Lv_xCS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");


}
*/

// Copy CS table to the device
void CrossSectionsManager::m_copy_cs_table_cpu2gpu() {

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











#endif
