// GGEMS Copyright (C) 2017

/*!
 * \file data_procesing.cuh
 * \brief Functions allowing processing data (filtering, resampling, etc.)
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date May 4 2017
 *
 *
 *
 */

#ifndef PROCESSING_CUH
#define PROCESSING_CUH

#include "global.cuh"
#include "voxelized.cuh"
#include "image_io.cuh"


// DataProcessing
namespace DataProcessing
{
/*
    f32 *mean( f32* input,  ui32 nx, ui32 ny, ui32 nz, ui32 w_size );
    f32 *median( f32* input,  ui32 nx, ui32 ny, ui32 nz, ui32 w_size );
    f32 *adaptive_median( f32* input, ui32 nx, ui32 ny, ui32 nz, ui32 w_size, ui32 w_size_max );
    f32 *resampling_lanczos3( f32* input, ui32 nx, ui32 ny, ui32 nz, ui32 new_nx, ui32 new_ny, ui32 new_nz );
    f32 *cropping_vox_around_center( f32* input, ui32 nx, ui32 ny, ui32 nz,
                                     i32 xmin, i32 xmax, i32 ymin, i32 ymax, i32 zmin, i32 zmax );
    void capping_values( f32* input, ui32 nx, ui32 ny, ui32 nz, f32 val_min, f32 val_max );
    void scale_values( f32* input, ui32 nx, ui32 ny, ui32 nz, f32 val_scale );
*/

    /*!
     * \fn VoxVolumeData<f32> *new_volume_zeros( VoxVolumeData<f32> *input )
     * \brief Initialized a new volume to zeros based on an existing volume (same spacings, dimensions, etc.)
     * \param input existing volume
     * \return  New volume initialized to zeros
     */
    VoxVolumeData<f32> *new_volume_zeros( const VoxVolumeData<f32> *input );


    /*!
     * \fn VoxVolumeData<f32> *filter_mean( VoxVolumeData<f32> *input, ui32 w_size )
     * \brief Mean filter (3D)
     * \param input input voxelized volume (float)
     * \param w_size window size (odd value)
     * \return New voxelized volume after filtering
     */
    VoxVolumeData<f32> *filter_mean( const VoxVolumeData<f32> *input, ui32 w_size );

    /*!
     * \fn VoxVolumeData<f32> *filter_median( VoxVolumeData<f32> *input, ui32 w_size )
     * \brief Median filter (3D)
     * \param input input voxelized volume (float)
     * \param w_size window size (odd value)
     * \return New voxelized volume after filtering
     */
    VoxVolumeData<f32> *filter_median( const VoxVolumeData<f32> *input, ui32 w_size );

    /*!
     * \fn VoxVolumeData<f32> *filter_adaptive_median( VoxVolumeData<f32> *input, ui32 w_size, ui32 w_size_max )
     * \brief Adaptive median filter (3D)
     * \param input input voxelized volume (float)
     * \param w_size window size (odd value)
     * \param w_size_maw maximum window size (odd value)
     * \return New voxelized volume after filtering
     */
    VoxVolumeData<f32> *filter_adaptive_median( const VoxVolumeData<f32> *input, ui32 w_size, ui32 w_size_max );

    /*!
     * \fn VoxVolumeData<f32> *resampling_lanczos3( VoxVolumeData<f32> *input, ui32 new_nx, ui32 new_ny, ui32 new_nz )
     * \brief Volume resampling using Lanczos3 method
     * \param input input voxelized volume (float)
     * \param new_nx new size of the volume along x-axis (number of voxels)
     * \param new_ny new size of the volume along y-axis (number of voxels)
     * \param new_nz new size of the volume along z-axis (number of voxels)
     * \return New voxelized volume after resampling
     */
    VoxVolumeData<f32> *resampling_lanczos3( const VoxVolumeData<f32> *input, ui32 new_nx, ui32 new_ny, ui32 new_nz );


    /*!
     * \fn f32xyzw mask_info( VoxVolumeData<f32> *vol, std::string mask_name, ui32 id_mask = 1 )
     * \brief Function that use a mask to return values information of a volume
     * \param input voxelized volume to process
     * \param mask_name mask filename
     * \param id_mask mask volume may contains different organ label, then select the label of the targeted mask
     * \return Min, max, mean, and std values of the voxelized volume
     */
    f32xyzw mask_info( const VoxVolumeData<f32> *input, std::string mask_name, ui32 id_mask = 1 );
}


#endif
