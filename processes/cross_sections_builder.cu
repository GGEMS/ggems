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

        Photon_CS_table.E = (float*)malloc(tot_elt_mem);
        Photon_CS_table.Compton_Std_CS = (float*)malloc(tot_elt_mem);
        Photon_CS_table.PhotoElectric_Std_CS = (float*)malloc(tot_elt_mem);
        Photon_CS_table.Rayleigh_Lv_CS = (float*)malloc(tot_elt_mem);
        Photon_CS_table.Rayleigh_Lv_SF = (float*)malloc(tot_elt_mem);

        Photon_CS_table.E_min = min_E;
        Photon_CS_table.E_max = max_E;
        Photon_CS_table.nb_bins = nbin;
        Photon_CS_table.nb_mat = materials.nb_materials;

        ////////////// TODO - THIS IS NOT NEED //////////////////////////
        i=0; while (i < tot_elt) {
            Photon_CS_table.E[i] = 0.0f;
            Photon_CS_table.Compton_Std_CS[i] = 0.0f;
            Photon_CS_table.PhotoElectric_Std_CS[i] = 0.0f;
            Photon_CS_table.Rayleigh_Lv_CS[i] = 0.0f;
            Photon_CS_table.Rayleigh_Lv_SF[i] = 0.0f;
            ++i;
        }
        //////////////////////////////////////////////////////////
    }

    // idem for e- table - TODO
    // if (there_is_electron) {
    // ...
    // ...

    // If Rayleigh scattering, load information once from G4 EM data library
    float *g4_ray_cs = NULL;
    float *g4_ray_sf = NULL;
    if (parameters.physics_list[PHOTON_RAYLEIGH]) {
        g4_ray_cs = Rayleigh_CS_Livermore_load_data();
        g4_ray_sf = Rayleigh_SF_Livermore_load_data();
    }

    // Get CS for each material, energy bin and phys effect
    unsigned int imat=0;
    unsigned int abs_index;
    while (imat < materials.nb_materials) {

        // for each energy bin
        i=0; while (i < nbin) {

            // absolute index to store data within the table
            abs_index = imat*materials.nb_materials + i;

            // store energy value
            if (there_is_photon) Photon_CS_table.E[abs_index] = E_scale[i];
            //if (there_is_electron) Electron_CS_table.E[abs_index] = E_scale[i];

            // for each phys effect
            if (parameters.physics_list[PHOTON_COMPTON]) {
                Photon_CS_table.Compton_Std_CS[abs_index] = Compton_CS_standard(materials, imat, E_scale[i]);
            }
            if (parameters.physics_list[PHOTON_PHOTOELECTRIC]) {
                Photon_CS_table.PhotoElectric_Std_CS[abs_index] = PhotoElec_CS_standard(materials, imat, E_scale[i]);
            }
            if (parameters.physics_list[PHOTON_RAYLEIGH]) {
                Photon_CS_table.Rayleigh_Lv_CS[abs_index] = Rayleigh_CS_Livermore(materials, g4_ray_cs,
                                                                                  imat, E_scale[i]);
                Photon_CS_table.Rayleigh_Lv_SF[abs_index] = Rayleigh_SF_Livermore(g4_ray_sf,
                                                                                  imat, E_scale[i]);
            }

            // TODO
            // idem with Electron_CS_table

            ++i;
        } // i

        ++imat;
    } // imat

    // Free mem
    free(E_scale);


}




#endif
