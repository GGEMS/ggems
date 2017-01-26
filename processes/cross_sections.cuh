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

        void initialize(const MaterialsData *h_materials, const GlobalSimulationParametersData *h_parameters);
        
        // CS data
        PhotonCrossSectionData *h_photon_CS;
        PhotonCrossSectionData *d_photon_CS;
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

        const GlobalSimulationParametersData *mh_parameters;
        const MaterialsData *mh_materials;
};

#endif
