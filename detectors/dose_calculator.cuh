// GGEMS Copyright (C) 2015

/*!
 * \file dose_calculator.cuh
 * \brief
 * \author Y. Lemaréchal
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 * \todo a) Need to add the output function for the dose map while ImageReader will be ready
 * \todo b) Fix: change ox, oy, oz to off_x, off_y, off_z
 *
 */

#ifndef DOSE_CALCULATOR_CUH
#define DOSE_CALCULATOR_CUH

#include "global.cuh"
#include "particles.cuh"
#include "vector.cuh"
#include "image_reader.cuh"

/**
 * \struct Dosimetry
 * \brief Dosimetry structure
 *
 * Structure where dosimetry parameters are store during the simulation (edep and edep squared). The size of the dosimetry volume is the same as the voxelised volume
 * \param edep Energy deposited inside the volume
 * \param dose Dose deposited inside the volume
 * \param edep_Squared_value Energy squared deposited inside the volume
 * \param uncertainty Uncertainty associated with the Energy deposited inside the volume
 * \param nb_of_voxels Number of voxels inside the volume, and also the size of the dosimetrics array
 **/
struct DoseData
{
    // Data
    f32 * edep;
    f32 * dose;
    f32 * edep_squared;
    ui32 * number_of_hits;        
    f32 * uncertainty;
    
    // Number of voxels per dimension
    ui32 nx;
    ui32 ny;
    ui32 nz;
    
    // Voxel size per dimension
    f32 spacing_x;
    f32 spacing_y;
    f32 spacing_z;
    
    // Offset
    f32 ox;
    f32 oy;
    f32 oz;
    
    ui32 nb_of_voxels;
};

// Struct that handle CPU&GPU data
struct Dose
{
    DoseData data_h;
    DoseData data_d;
};

// Dose functions
__host__ __device__ void dose_record_standard(DoseData dose, f32 Edep, f32xyz pos);

// Class
class DoseCalculator {

    public:
        DoseCalculator();
        ~DoseCalculator();

        // Setting
        void set_size_in_voxel(ui32 x, ui32 y, ui32 z);
        void set_voxel_size(f32 sx, f32 sy, f32 sz);
        void set_offset(f32 ox, f32 oy, f32 oz);
        
        void initialize(GlobalSimulationParameters params);

        Dose dose;

    private :
        bool m_check_mandatory();
        void m_cpu_malloc_dose();
        void m_gpu_malloc_dose();
        void m_copy_dose_cpu2gpu();



        
};




#endif
