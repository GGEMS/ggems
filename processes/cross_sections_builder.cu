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

#ifndef CROSS_SECTIONS_BUILDER_CU
#define CROSS_SECTIONS_BUILDER_CU
#include "cross_sections_builder.cuh"

//// CrossSectionsBuilder class ////////////////////////////////////////////////////

CrossSectionsBuilder::CrossSectionsBuilder() {
    photon_CS_table.nb_bins = 0;
    photon_CS_table.nb_mat = 0;
}

// Build cross sections table according material, physics effects and particles
void CrossSectionsBuilder::build_table(MaterialsTable materials, GlobalSimulationParameters parameters) {

    // Read parameters
    unsigned int nbin = parameters.cs_table_nbins;
    float min_E = parameters.cs_table_min_E;
    float max_E = parameters.cs_table_max_E;

    // First thing first, sample energy following the number of bins
    float slope = log(max_E / min_E);
    unsigned int i = 0;
    float *E_scale = (float*)malloc(nbin * sizeof(float));
    while (i < nbin) {
        E_scale[i] = min_E * exp( slope * ((float)i/((float)nbin-1)) ) * MeV;
        ++i;
    }

    // Find if there are photon and electron in this simulation;
    char there_is_photon = parameters.physics_list[PHOTON_COMPTON] ||
                           parameters.physics_list[PHOTON_PHOTOELECTRIC] ||
                           parameters.physics_list[PHOTON_RAYLEIGH];
    //char there_is_electron = parameters.physics_list[ELECTRON_IONISATION] ||
    //                         parameters.physics_list[ELECTRON_BREMSSTRAHLUNG] ||
    //                         parameters.physics_list[ELECTRON_MSC];

    // Then init data
    unsigned int tot_elt = materials.nb_materials*nbin;
    unsigned int tot_elt_mem = tot_elt * sizeof(float);

    // Photon CS table if need
    if (there_is_photon) {

        photon_CS_table.E_CS = (float*)malloc(tot_elt_mem);
        photon_CS_table.Compton_Std_CS = (float*)malloc(tot_elt_mem);
        photon_CS_table.PhotoElectric_Std_CS = (float*)malloc(tot_elt_mem);
        photon_CS_table.Rayleigh_Lv_CS = (float*)malloc(tot_elt_mem);
        photon_CS_table.Rayleigh_Lv_SF = (float*)malloc(nbin * 101 * sizeof(float)); // 100 Z elements,
                                                                                     // starting from index 1
        photon_CS_table.Rayleigh_Lv_xCS = (float*)malloc(nbin * 101 * sizeof(float)); // 100 Z elements,
                                                                                     // starting from index 1
        photon_CS_table.E_SF = (float*)malloc(nbin * 101 * sizeof(float)); // 100 Z elements,
                                                                           // starting from index 1

        photon_CS_table.E_min = min_E;
        photon_CS_table.E_max = max_E;
        photon_CS_table.nb_bins = nbin;
        photon_CS_table.nb_mat = materials.nb_materials;

        // Init value
        i=0; while (i < tot_elt) {
            photon_CS_table.E_CS[i] = 0.0f;
            photon_CS_table.Compton_Std_CS[i] = 0.0f;
            photon_CS_table.PhotoElectric_Std_CS[i] = 0.0f;
            photon_CS_table.Rayleigh_Lv_CS[i] = 0.0f;

            ++i;
        }
        i=0; while (i < (101*nbin)) { // 100 Z element starting from index 1
            photon_CS_table.Rayleigh_Lv_SF[i] = 0.0f;
            photon_CS_table.Rayleigh_Lv_xCS[i] = 0.0f;
            photon_CS_table.E_SF[i] = 0.0f;
            ++i;
        }

    }

    // idem for e- table - TODO
    // if (there_is_electron) {
    // ...
    // ...

    // If Rayleigh scattering, load information once from G4 EM data library
    float *g4_ray_cs = NULL;
    float *g4_ray_sf = NULL;
    char *flag_Z = NULL;
    if (parameters.physics_list[PHOTON_RAYLEIGH]) {
        g4_ray_cs = Rayleigh_CS_Livermore_load_data();
        g4_ray_sf = Rayleigh_SF_Livermore_load_data();
        // use to flag is scatter factor are already defined for a given Z
        flag_Z = (char*)malloc(101*sizeof(char));
        i=0; while(i<101) {flag_Z[i]=0; ++i;}
    }

    // Get CS for each material, energy bin and phys effect
    unsigned int imat=0;
    unsigned int abs_index;
    while (imat < materials.nb_materials) {

        // for each energy bin
        i=0; while (i < nbin) {

            // absolute index to store data within the table
            abs_index = imat*nbin + i;

            // store energy value
            if (there_is_photon) photon_CS_table.E_CS[abs_index] = E_scale[i];
            //if (there_is_electron) Electron_CS_table.E[abs_index] = E_scale[i];

            // for each phys effect
            if (parameters.physics_list[PHOTON_COMPTON]) {
                photon_CS_table.Compton_Std_CS[abs_index] = Compton_CS_standard(materials, imat, E_scale[i]);
            }
            if (parameters.physics_list[PHOTON_PHOTOELECTRIC]) {
                photon_CS_table.PhotoElectric_Std_CS[abs_index] = PhotoElec_CS_standard(materials, imat, E_scale[i]);
            }
            if (parameters.physics_list[PHOTON_RAYLEIGH]) {
                photon_CS_table.Rayleigh_Lv_CS[abs_index] = Rayleigh_CS_Livermore(materials, g4_ray_cs,
                                                                                  imat, E_scale[i]);
            }

            // TODO
            // idem with Electron_CS_table

            ++i;
        } // i

        // Special case for Rayleigh where scatter factor and CS are needed for each Z
        if (parameters.physics_list[PHOTON_RAYLEIGH]) {
            unsigned int iZ, Z;
            // This table compute scatter factor for each Z (only for Z which were not already defined)
            iZ=0; while (iZ < materials.nb_elements[imat]) {
                Z = materials.mixture[materials.index[imat]+iZ];

                // If for this Z nothing was already calculated
                if (!flag_Z[Z]) {
                    flag_Z[Z] = 1;

                    // for each energy bin
                    i=0; while (i < nbin) {
                        // absolute index to store data within the table
                        abs_index = Z*nbin + i;
                        photon_CS_table.Rayleigh_Lv_SF[abs_index] = Rayleigh_SF_Livermore(g4_ray_sf,
                                                                                          E_scale[i],
                                                                                          Z);
                        // TODO
                        // add: mat.atom_num_dens[index+i] * CSPA
                        // add: CDF (starting from 0) -> xsec +=

                        photon_CS_table.Rayleigh_Lv_xCS[abs_index] = Rayleigh_CSPA_Livermore(g4_ray_cs,
                                                                                             E_scale[i],
                                                                                             Z);

                        photon_CS_table.E_SF[abs_index] = E_scale[i];

                        ++i;
                    } // i
                } // flag_Z
                ++iZ;

            } // iZ
        } // if

        ++imat;
    } // imat

    // Free mem
    free(E_scale);
    free(flag_Z);
}

// Print CS talbe (for debugging)
void CrossSectionsBuilder::print() {

    unsigned int imat, iE, abs_index;


    printf("::::::::::::::::::::::::::::::::::::::::::::\n");
    printf("::::::::::::::::: Gamma ::::::::::::::::::::\n");
    printf("::::::::::::::::::::::::::::::::::::::::::::\n\n");

    printf("==== Compton Standard CS ====\n\n");

    imat=0; while (imat < photon_CS_table.nb_mat) {
        printf("## Material %i\n", imat);
        iE=0; while (iE < photon_CS_table.nb_bins) {
            abs_index = imat*photon_CS_table.nb_bins + iE;
            printf("E %e CS %e\n", photon_CS_table.E_CS[abs_index],
                                   photon_CS_table.Compton_Std_CS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");

    printf("==== Photoelectric Standard CS ====\n");

    imat=0; while (imat < photon_CS_table.nb_mat) {
        printf("## Material %i\n", imat);
        iE=0; while (iE < photon_CS_table.nb_bins) {
            abs_index = imat*photon_CS_table.nb_bins + iE;
            printf("E %e CS %e\n", photon_CS_table.E_CS[abs_index],
                                   photon_CS_table.PhotoElectric_Std_CS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");

    printf("==== Rayleigh Livermore CS ====\n");

    imat=0; while (imat < photon_CS_table.nb_mat) {
        printf("## Material %i\n", imat);
        iE=0; while (iE < photon_CS_table.nb_bins) {
            abs_index = imat*photon_CS_table.nb_bins + iE;
            printf("E %e CS %e\n", photon_CS_table.E_CS[abs_index],
                                   photon_CS_table.Rayleigh_Lv_CS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");

    printf("==== Rayleigh Livermore SF ====\n");

    imat=0; while (imat < 101) {
        printf("## Z %i\n", imat);
        iE=0; while (iE < photon_CS_table.nb_bins) {
            abs_index = imat*photon_CS_table.nb_bins + iE;
            printf("E %e SF %e CS %e\n", photon_CS_table.E_SF[abs_index],
                                         photon_CS_table.Rayleigh_Lv_SF[abs_index],
                                         photon_CS_table.Rayleigh_Lv_xCS[abs_index]);
            ++iE;
        } // iE
        printf("\n");
        ++imat;
    } // imat
    printf("\n");


}




#endif
