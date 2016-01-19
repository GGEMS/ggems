// GGEMS Copyright (C) 2015

/*!
 * \file cross_sections.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef CROSS_SECTIONS_CUH
#define CROSS_SECTIONS_CUH

#include "materials.cuh"
#include "global.cuh"
#include "photon.cuh"
#include "electron.cuh"

// CS class
class CrossSections {
    public:
        CrossSections();

        void initialize(Materials materials, GlobalSimulationParameters parameters);
        void print();       
        
        // Data for photon
        PhotonCrossSection photon_CS;       // CPU & GPU
        ElectronsCrossSection electron_CS;  // CPU & GPU
        
        // Data for electron TODO
        //ElectronCrossSection Electron_CS;
        //ElectronsCrossSectionTable get_electron_data_h(){return electronCSTable->data_h;}
        //ElectronsCrossSectionTable get_electron_data_d(){return electronCSTable->data_d;}
        ElectronCrossSection *electronCSTable;
    private:        
        bool m_check_mandatory();
        void m_build_table(Materials materials, GlobalSimulationParameters parameters);
        void m_copy_cs_table_cpu2gpu();

        // Electron CS data
        
        GlobalSimulationParameters params;
        GlobalSimulationParameters *parameters;
        
        i8 there_is_photon;
        i8 there_is_electron;

};

#endif
