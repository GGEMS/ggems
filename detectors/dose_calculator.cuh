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
 * \todo c) Dose uncertainty calculation, this part can also be implemented on the device side (GPU)
 * \todo d) Check if density in really in g/mm3
 */

#ifndef DOSE_CALCULATOR_CUH
#define DOSE_CALCULATOR_CUH

#include "global.cuh"
#include "particles.cuh"
#include "vector.cuh"
#include "image_reader.cuh"
#include "voxelized.cuh"
#include "materials.cuh"

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
    f64 * edep;
    f64 * dose;
    f64 * edep_squared;
    ui32 * number_of_hits;        
    f64 * uncertainty;
    
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
__host__ __device__ void dose_record_standard(DoseData dose, f32 Edep, f32 px, f32 py, f32 pz);

// Class
class DoseCalculator {

    public:
        DoseCalculator();
        ~DoseCalculator();

        // Setting
        void set_size_in_voxel(ui32 x, ui32 y, ui32 z);
        void set_voxel_size(f32 sx, f32 sy, f32 sz);
        void set_offset(f32 ox, f32 oy, f32 oz);
        void set_voxelized_phantom(VoxelizedPhantom aphantom);
        void set_materials(Materials materials);
        void set_min_density(f32 min); // Min density to consider the dose calculation

        // Init
        void initialize(GlobalSimulationParameters params);

        // Dose calculation
        void calculate_dose_to_water();
        void calculate_dose_to_phantom();

        Dose dose;

    private :
        bool m_check_mandatory();
        void m_cpu_malloc_dose();
        void m_gpu_malloc_dose();

        void m_copy_dose_cpu2gpu();
        void m_copy_dose_gpu2cpu();

        VoxelizedPhantom m_phantom;
        Materials m_materials;
        bool m_flag_phantom;
        bool m_flag_materials;

        f32 m_dose_min_density;

        GlobalSimulationParameters m_params;



        
};




#endif
