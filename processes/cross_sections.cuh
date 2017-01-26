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
//#include "electron.cuh"

// CS class
class CrossSections {
    public:
        CrossSections();

        void initialize(Materials materials, GlobalSimulationParametersData *h_parameters);
        
        // CS data
        PhotonCrossSection photon_CS;       // CPU & GPU
//        ElectronsCrossSection electron_CS;  // CPU & GPU

    private:        
        bool m_check_mandatory();

        // For gamma
        void m_build_photon_table();
        void m_copy_photon_cs_table_cpu2gpu();

/*
        // For e-
        void m_build_electron_table();
        f32 m_get_electron_dedx( f32 energy, ui8 mat_id );
        void m_copy_electron_cs_table_cpu2gpu();
        void m_dump_electron_tables( std::string dirname );
*/
        ui32 m_nb_bins, m_nb_mat;

        GlobalSimulationParametersData *mh_parameters;
        MaterialsTable m_materials;
};

#endif
