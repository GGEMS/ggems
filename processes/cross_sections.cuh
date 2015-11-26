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

        void initialize(Materials materials, GlobalSimulationParameters parameters);
        //void print();

        // Data for photon
        PhotonCrossSection photon_CS;  // CPU & GPU

        // Data for electron TODO
        //ElectronCrossSection Electron_CS;

    private:        
        bool m_check_mandatory();
        void m_build_table(Materials materials, GlobalSimulationParameters parameters);
        void m_copy_cs_table_cpu2gpu();



};

#endif
