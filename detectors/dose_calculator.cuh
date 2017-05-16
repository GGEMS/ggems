// GGEMS Copyright (C) 2017

/*!
 * \file dose_calculator.cuh
 * \brief 
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.4
 * \date 02/12/2015
 * \date 26/02/2016, add volume of interest, change offset handling and fix many bugs - JB
 *
 * v0.4: JB - Change all structs and remove CPU exec
 *
 */

#ifndef DOSE_CALCULATOR_CUH
#define DOSE_CALCULATOR_CUH

#include "global.cuh"
#include "particles.cuh"
#include "vector.cuh"
#include "image_io.cuh"
#include "voxelized.cuh"
#include "materials.cuh"
#include "raytracing.cuh"
#include "fun.cuh"

struct DoseData
{
    // Data
    f64 *edep;    
    f64 *edep_squared;
    ui32 *number_of_hits;

    // Number of dosels per dimension
    ui32xyz nb_dosels;

    // Doxel size per dimension
    f32xyz dosel_size;

    // Inverse of dosel size (for calculation latter)
    f32xyz inv_dosel_size;

    // Offset
    f32xyz offset;

    // Volume Of Interest
    f32 xmin, xmax;
    f32 ymin, ymax;
    f32 zmin, zmax;

    // Tot number of dosels
    ui32 tot_nb_dosels;
    // Number of voxels per slice
    ui32 slice_nb_dosels;
};

// Dose functions
__host__ __device__ void dose_record_standard ( DoseData *dose, f32 Edep, f32 px, f32 py, f32 pz );
__host__ __device__ void dose_record_TLE ( DoseData *dose, f32 Edep, f32 px, f32 py, f32 pz,
                                           f32 length, f32 mu_en);

// Class
class DoseCalculator
{

public:
    DoseCalculator();
    ~DoseCalculator();

    // Setting
    //void set_size_in_voxel ( ui32 x, ui32 y, ui32 z );
    void set_dosel_size( f32 sx, f32 sy, f32 sz );
    void set_voi( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax);
    //void set_offset ( f32 ox, f32 oy, f32 oz );
    void set_voxelized_phantom( VoxelizedPhantom aphantom );
    void set_materials( Materials materials );
    void set_min_density( f32 min ); // Min density to consider the dose calculation

    // Updating
    void clear_deposition();

    // Getting
    VoxVolumeData<f32> * get_dose_map();
    VoxVolumeData<f32> * get_uncertainty_map();

    // Init
    void initialize( GlobalSimulationParametersData *h_params );

    // Dose calculation
    void calculate_dose_to_water();
    void calculate_dose_to_medium();
    void dose_scaling( f32 factor );

    void write( std::string filename = "dosimetry.mhd", std::string option = "all" );

    DoseData *h_dose;
    DoseData *d_dose;

private :
    bool m_check_mandatory();
    void m_uncertainty_calculation( ui32 dosel_id_x, ui32 dosel_id_y, ui32 dosel_id_z );
    void m_dose_to_water_calculation( ui32 dosel_id_x, ui32 dosel_id_y, ui32 dosel_id_z );
    void m_dose_to_phantom_calculation( ui32 dosel_id_x, ui32 dosel_id_y, ui32 dosel_id_z );
    void m_copy_dosemap_to_gpu();
    void m_copy_dosemap_to_cpu();    

    VoxelizedPhantom m_phantom;
    Materials m_materials;

    f32xyz m_dosel_size;
    f32xyz m_offset;
    ui32xyz m_nb_of_dosels;
    f32 m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax;

    f32 m_dose_min_density;

    GlobalSimulationParametersData *mh_params;

    f32 *m_dose_values;
    f32 *m_uncertainty_values;

    bool m_flag_dose_calculated;
    bool m_flag_uncertainty_calculated;
    bool m_flag_phantom;
    bool m_flag_materials;


};




#endif
