// GGEMS Copyright (C) 2017

/*!
 * \file dvh_calculator.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 19/04/2016
 *
 *
 */

#ifndef DVH_CALCULATOR_CU
#define DVH_CALCULATOR_CU

#include "dvh_calculator.cuh"

/// Class //////////////////////////////////////////////

DVHCalculator::DVHCalculator()
{
    m_dvh_calcualted = false;
    m_dose_min =  F32_MAX;
    m_dose_max = -F32_MAX;
    m_dose_total = 0.0;
    m_mean_dose = 0.0;
    m_std_dose = 0.0;
    m_nb_dosels = 0;
    m_nb_bins = 0;

    m_spacing_x = 0;
    m_spacing_y = 0;
    m_spacing_z = 0;

    m_dvh_bins = NULL;
    m_dvh_values = NULL;

}

DVHCalculator::~DVHCalculator()
{
    delete[] m_dvh_bins;
    delete[] m_dvh_values;
}

/// Main function //////////////////////////////////////////////

void DVHCalculator::compute_dvh_from_mask(VoxVolumeData<f32> *dosemap, std::string mask_name, ui32 id_mask, ui32 nb_of_bins)
{
    // First open the mask
//    f32xyz offset, voxsize;
//    ui32xyz nbvox;
//    f32* mask = ImageReader::load_mhd_image( mask_name, offset, voxsize, nbvox );

    ImageIO *image_io = new ImageIO;

    image_io->open( mask_name );

    f32 *mask = image_io->get_image_in_f32();
    //f32xyz offset = image_io->get_offset();
    //f32xyz voxsize = image_io->get_spacing();
    ui32xyz nbvox = image_io->get_size();
    delete image_io;

    // Check if mask and dosemap have the same size
    if ( nbvox.x != dosemap->nb_vox_x || nbvox.y != dosemap->nb_vox_y || nbvox.z != dosemap->nb_vox_z )
    {
        GGcerr << "DVH: Dosemap and mask have not the same size!" << GGendl;
        GGcerr << "     dose: " << dosemap->nb_vox_x << "x"
                                << dosemap->nb_vox_y << "x"
                                << dosemap->nb_vox_x << " "
               << "mask: " << nbvox.x << "x"
                           << nbvox.y << "x"
                           << nbvox.z << GGendl;
        return;
    }

    // Store some values
    m_spacing_x = dosemap->spacing_x;
    m_spacing_y = dosemap->spacing_y;
    m_spacing_z = dosemap->spacing_z;

    // Get min, max, mean, std, total dose and nb of voxels under the mask
    f32 val = 0.0;
    ui32 i = 0;
    m_nb_dosels = 0;
    m_dose_total = 0.0;
    while( i < dosemap->number_of_voxels )
    {
        if ( mask[ i ] == id_mask )
        {
            val = dosemap->values[ i ];

            m_dose_total += val;

            if ( val > m_dose_max ) m_dose_max = val;
            if ( val < m_dose_min ) m_dose_min = val;

            m_nb_dosels++;
        }

        i++;
    }
    m_mean_dose = m_dose_total / f32( m_nb_dosels );

    // Std
    m_std_dose = 0.0;
    i = 0; while( i < dosemap->number_of_voxels )
    {
        if ( mask[ i ] == id_mask )
        {
            val = dosemap->values[ i ];
            m_std_dose += (val - m_mean_dose)*(val - m_mean_dose);
        }

        i++;
    }
    m_std_dose /= f32( m_nb_dosels );
    m_std_dose  = sqrt( m_std_dose );

    /// Build dvh (histogram of dose) ///

    // Init mem
    m_nb_bins = nb_of_bins;

    m_dvh_bins = new f32[ m_nb_bins ];
    m_dvh_values = new f32[ m_nb_bins ];
    f32 *buffer_dvh_values = new f32[ m_nb_bins ];

    i = 0; while( i < m_nb_bins )
    {
        m_dvh_bins[ i ] = 0.0;
        m_dvh_values[ i ] = 0.0;
        buffer_dvh_values[ i ] = 0.0;
        i++;
    }

    f32 di = ( m_dose_max - m_dose_min ) / f32( m_nb_bins - 1 );

    // Get Histo
    ui32 posi;
    i = 0; while( i < dosemap->number_of_voxels )
    {
        if ( mask[ i ] == id_mask )
        {
            posi = ( dosemap->values[ i ] - m_dose_min ) / di;

#ifdef DEBUG
            assert( posi < m_nb_bins );
#endif

            buffer_dvh_values[ posi ]++;
        }

        i++;
    }

    // Get bin values
    i = 0; while( i < m_nb_bins )
    {
        m_dvh_bins[ i ] = m_dose_min + i*di;
        i++;
    }

//    // Compute the reverse CDF to get the final DVH
//    m_dvh_values[ m_nb_bins-1 ] = buffer_dvh_values[ m_nb_bins-1 ];
//    i = m_nb_bins - 2;
//    while (i >= 0)
//    {
//        m_dvh_values[ i ] = m_dvh_values[ i+1 ] + buffer_dvh_values[ i ];
//        i--;
//    }

    // Compute the CDF to get the final DVH
    m_dvh_values[ 0 ] = buffer_dvh_values[ 0 ];
    i = 1;
    while (i < m_nb_bins )
    {
        m_dvh_values[ i ] = m_dvh_values[ i-1 ] + buffer_dvh_values[ i ];
        i++;
    }

    // Normalize DVH to 100% of volume
    f32 max_vol = m_dvh_values[ m_nb_bins-1 ];
    i = 0; while( i < m_nb_bins )
    {
        m_dvh_values[ i++ ] /= (0.01 * max_vol); // in % of vol max
    }

    // Free mem
    delete[] buffer_dvh_values;

    // DVH calculation completed
    m_dvh_calcualted = true;

}

/// Getting ///////////////////////////////////////////////////////////:

f32 DVHCalculator::get_max_dose()
{
    if ( !m_dvh_calcualted )
    {
        GGcerr << "First you need to compute DVH values from dose map!" << GGendl;
        return 0.0;
    }

    return m_dose_max;
}

f32 DVHCalculator::get_min_dose()
{
    if ( !m_dvh_calcualted )
    {
        GGcerr << "First you need to compute DVH values from dose map!" << GGendl;
        return 0.0;
    }

    return m_dose_min;
}

f32 DVHCalculator::get_total_dose()
{
    if ( !m_dvh_calcualted )
    {
        GGcerr << "First you need to compute DVH values from dose map!" << GGendl;
        return 0.0;
    }

    return m_dose_total;
}

f32 DVHCalculator::get_mean_dose()
{
    if ( !m_dvh_calcualted )
    {
        GGcerr << "First you need to compute DVH values from dose map!" << GGendl;
        return 0.0;
    }

    return m_mean_dose;
}

f32 DVHCalculator::get_std_dose()
{
    if ( !m_dvh_calcualted )
    {
        GGcerr << "First you need to compute DVH values from dose map!" << GGendl;
        return 0.0;
    }

    return m_std_dose;
}

f32 DVHCalculator::get_total_volume_size()
{
    return m_nb_dosels * m_spacing_x*m_spacing_y*m_spacing_z;
}

f32 DVHCalculator::get_dose_from_volume_percent( f32 volume_percent )
{
    // Check value
    if( volume_percent < 0 || volume_percent > 100 )
    {
        GGwarn << "DVH: volume_percent value must be between 0 and 100, '" << volume_percent << "' given!" << GGendl;
        return 0.0;
    }

    if ( !m_dvh_calcualted )
    {
        GGcerr << "First you need to compute DVH values from dose map!" << GGendl;
        return 0.0;
    }

    ui32 index = binary_search( volume_percent, m_dvh_values, m_nb_bins );

    if ( index == 0 )
    {
        return m_dvh_bins[ index ];
    }
    else
    {
        return linear_interpolation( m_dvh_values[ index-1 ],  m_dvh_bins[ index-1 ],
                                     m_dvh_values[ index ],    m_dvh_bins[ index ],
                                     volume_percent );
    }
}

/// Priting ///////////////////////////////////////////////////////////////

void DVHCalculator::print_dvh()
{

    if ( !m_dvh_calcualted )
    {
        GGwarn << "First you need to compute DVH values from dose map!" << GGendl;
    }

    GGcout << "=== DVH ===" << GGendl;
    GGcout << "dose - volume" << GGendl;
    ui32 i = 0;
    while (i < m_nb_bins )
    {
        GGcout << m_dvh_bins[ i ] << " " << m_dvh_values[ i ] << GGendl;
        ++i;
    }

}










#endif
