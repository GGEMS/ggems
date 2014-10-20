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

#ifndef CROSS_SECTIONS_BUILDER_CUH
#define CROSS_SECTIONS_BUILDER_CUH


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <algorithm>
//#include <vector>

#include "../geometry/materials.cuh"
#include "../global/global.cuh"
#include "photon.cuh"

// Cross section table for photon particle
struct PhotonCrossSectionTable{
    float* E_CS;                  // n*k
    float* Compton_Std_CS;        // n*k
    float* PhotoElectric_Std_CS;  // n*k
    float* Rayleigh_Lv_CS;        // n*k

    float* E_SF;                  // n*101
    float* Rayleigh_Lv_SF;        // n*101 (Nb of Z)
    float* Rayleigh_Lv_xCS;       // n*101 (Nb of Z)

    float E_min;
    float E_max;
    unsigned int nb_bins;         // n
    unsigned int nb_mat;          // k
};

// CS class
class CrossSectionsBuilder {
    public:
        CrossSectionsBuilder();
        void build_table(MaterialsTable materials, GlobalSimulationParameters parameters);
        void print();

        // Data for photon
        PhotonCrossSectionTable photon_CS_table;
        //ElectronCrossSectionTable Electron_CS_table;

    private:



};

#endif
