// GGEMS Copyright (C) 2015

/*!
 * \file em_calculator.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 31/10/2016
 *
 *
 */

#ifndef EM_CALCULATOR_CUH
#define EM_CALCULATOR_CUH

#include "global.cuh"
#include "materials.cuh"
#include "txt_reader.cuh"
#include "cross_sections.cuh"
#include "particles.cuh"
#include "photon_navigator.cuh"
#include "vector.cuh"

// Class
class EmCalculator
{

public:
    EmCalculator();
    ~EmCalculator();

    void initialize( std::string materials_db_filename );

    void compute_photon_tracking_uncorrelated_model(std::string mat_name, ui32 nb_samples,
                                                    f32 min_energy, f32 max_energy, ui32 nb_energy_bins,
                                                    f32 max_dist, f32 max_edep, ui32 nb_bins);

    void compute_photon_tracking_correlated_model(std::string mat_name, ui32 nb_samples,
                                                  f32 min_energy, f32 max_energy, ui32 nb_energy_bins,
                                                  f32 max_step, f32 max_substep, ui32 nb_step_bins, ui32 nb_theta_bins);

private:
    std::vector< std::string > m_get_all_materials_name( std::string filename );

private:
    Materials m_materials;
    CrossSections m_cross_sections;
    ParticleManager m_part_manager;
    GlobalSimulationParameters m_params;
    std::vector< std::string > m_mat_names_db;

};




#endif
