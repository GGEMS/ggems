// GGEMS Copyright (C) 2015

/*!
 * \file dvh_calculator.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 19/04/2016
 *
 *
 */

#ifndef DVH_CALCULATOR_CUH
#define DVH_CALCULATOR_CUH

#include "global.cuh"
#include "voxelized.cuh"
#include "image_reader.cuh"

// Class
class DVHCalculator
{

public:
    DVHCalculator();
    ~DVHCalculator();

    void compute_dvh_from_mask( VoxVolumeData<f32> dosemap, std::string mask_name, ui32 id_mask = 1, ui32 nb_of_bins = 100 );

    f32 get_dose_from_volume_percent( f32 volume_percent );
    f32 get_total_volume_size();
    f32 get_total_dose();
    f32 get_max_dose();
    f32 get_min_dose();




private:

    bool m_dvh_calcualted;

    f32 m_dose_min, m_dose_max, m_dose_total;
    f32 m_spacing_x, m_spacing_y, m_spacing_z;
    ui32 m_nb_dosels, m_nb_bins;

    f32 *m_dvh_bins;
    f32 *m_dvh_values;


};




#endif
