// GGEMS Copyright (C) 2015

#ifndef CROSS_SECTIONS_BUILDER_CUH
#define CROSS_SECTIONS_BUILDER_CUH

#include "materials.cuh"
#include "global.cuh"
#include "photon.cuh"

// CS class
class CrossSectionsBuilder {
    public:
        CrossSectionsBuilder();
        void build_table(MaterialsTable materials, GlobalSimulationParameters parameters);       
        void copy_cs_table_cpu2gpu();
        void print();

        // Data for photon
        PhotonCrossSectionTable photon_CS_table;   // CPU
        PhotonCrossSectionTable dphoton_CS_table;  // GPU

        // Data for electron TODO
        //ElectronCrossSectionTable Electron_CS_table;

    private:



};

#endif
