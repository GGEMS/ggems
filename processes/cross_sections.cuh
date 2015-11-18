// GGEMS Copyright (C) 2015

#ifndef CROSS_SECTIONS_CUH
#define CROSS_SECTIONS_CUH

#include "materials.cuh"
#include "global.cuh"
#include "photon.cuh"


// CS class
class CrossSectionsManager {
    public:
        CrossSectionsManager();

        void initialize(MaterialsTable materials, GlobalSimulationParameters parameters);
        void print();

        // Data for photon
        PhotonCrossSectionTable photon_CS_table_h;  // CPU
        PhotonCrossSectionTable photon_CS_table_d;  // GPU

        // Data for electron TODO
        //ElectronCrossSectionTable Electron_CS_table;

    private:        
        bool m_check_mandatory();
        void m_build_table(MaterialsTable materials, GlobalSimulationParameters parameters);
        void m_copy_cs_table_cpu2gpu();



};

#endif
