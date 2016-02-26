// GGEMS Copyright (C) 2015

/*!
 * \file dose_calculator.cuh
 * \brief
 * \author Y. Lemaréchal <yannick.lemarechal@univ-brest.fr>
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 */

#ifndef DOSE_CALCULATOR_CUH
#define DOSE_CALCULATOR_CUH

#include "global.cuh"
#include "particles.cuh"
#include "vector.cuh"
#include "image_reader.cuh"
#include "voxelized.cuh"
#include "materials.cuh"


struct DoseData
{
    // Data
    f64 *edep;
    f64 *dose;
    f64 *edep_squared;
    ui32 *number_of_hits;
    f64 *uncertainty;

    // Number of voxels per dimension
    ui32xyz nb_doxels;

    // Voxel size per dimension
    f32xyz doxel_size;

    // Offset
    f32xyz offset;

    // Volume Of Interest
    f32 xmin, xmax;
    f32 ymin, ymax;
    f32 zmin, zmax;

    ui32 tot_nb_doxels;
};

// Struct that handle CPU&GPU data
struct Dose
{
    DoseData data_h;
    DoseData data_d;
};

// Dose functions
__host__ __device__ void dose_record_standard ( DoseData &dose, f32 Edep, f32 px, f32 py, f32 pz );

// Class
class DoseCalculator
{

public:
    DoseCalculator();
    ~DoseCalculator();

    // Setting
    //void set_size_in_voxel ( ui32 x, ui32 y, ui32 z );
    void set_doxel_size ( f32 sx, f32 sy, f32 sz );
    void set_voi ( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax);
    //void set_offset ( f32 ox, f32 oy, f32 oz );
    void set_voxelized_phantom ( VoxelizedPhantom aphantom );
    void set_materials ( Materials materials );
    void set_min_density ( f32 min ); // Min density to consider the dose calculation

    // Init
    void initialize ( GlobalSimulationParameters params );

    // Dose calculation
    void calculate_dose_to_water();
    void calculate_dose_to_phantom();

    void write ( std::string filename = "dosimetry.mhd" );

    Dose dose;

private :
    bool m_check_mandatory();
    void m_cpu_malloc_dose();
    void m_gpu_malloc_dose();

    void m_copy_dose_cpu2gpu();
    void m_copy_dose_gpu2cpu();

    VoxelizedPhantom m_phantom;
    Materials m_materials;

    ui32xyz m_doxel_size;
    f32xyz m_offset;
    ui32xyz m_nb_of_doxels;
    f32 m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax;

    f32 m_dose_min_density;

    GlobalSimulationParameters m_params;

    bool m_flag_dose_calculated;


};




#endif
