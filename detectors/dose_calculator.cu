// GGEMS Copyright (C) 2015

/*!
 * \file dose_calculator.cu
 * \brief
 * \author Y. Lemaréchal
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 *
 *
 */

#ifndef DOSE_CALCULATOR_CU
#define DOSE_CALCULATOR_CU

#include "dose_calculator.cuh"

/// CPU&GPU functions
__host__ __device__ void dose_record_standard(DoseData dose, f32 Edep, f32xyz pos) {

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / dose.spacing_x;
    ivoxsize.y = 1.0 / dose.spacing_y;
    ivoxsize.z = 1.0 / dose.spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32( (pos.x+dose.ox) * ivoxsize.x );
    index_phantom.y = ui32( (pos.y+dose.oy) * ivoxsize.y );
    index_phantom.z = ui32( (pos.z+dose.oz) * ivoxsize.z );
    index_phantom.w = index_phantom.z*dose.nx*dose.ny
                      + index_phantom.y*dose.nx
                      + index_phantom.x; // linear index

    // Score dosemap
    ggems_atomic_add(dose.edep, index_phantom.w, Edep);
    ggems_atomic_add(dose.edep_squared, index_phantom.w, Edep*Edep);
    ggems_atomic_add(dose.number_of_hits, index_phantom.w, ui32(1));

}

/// Class
DoseCalculator::DoseCalculator()
{

    dose.data_h.nx = 0;
    dose.data_h.ny = 0;
    dose.data_h.nz = 0;
    
    // Voxel size per dimension
    dose.data_h.spacing_x = 0.0;
    dose.data_h.spacing_y = 0.0;
    dose.data_h.spacing_z = 0.0;
    
    // Offset
    dose.data_h.ox = 0.0;
    dose.data_h.oy = 0.0;
    dose.data_h.oz = 0.0;
        
    dose.data_h.edep = NULL;
    dose.data_h.dose = NULL;
    dose.data_h.edep_squared = NULL;
    dose.data_h.number_of_hits = NULL;
    
    dose.data_h.uncertainty = NULL;
}

DoseCalculator::~DoseCalculator()
{

    delete [] dose.data_h.edep ;
    delete [] dose.data_h.dose ;
    delete [] dose.data_h.edep_squared ;
    delete [] dose.data_h.number_of_hits ;
    delete [] dose.data_h.uncertainty ;

}

/// Setting
void DoseCalculator::set_size_in_voxel(ui32 nx, ui32 ny, ui32 nz) {
    dose.data_h.nx = nx;
    dose.data_h.ny = ny;
    dose.data_h.nz = nz;
}

void DoseCalculator::set_voxel_size(f32 sx, f32 sy, f32 sz) {
    dose.data_h.spacing_x = sx;
    dose.data_h.spacing_y = sy;
    dose.data_h.spacing_z = sz;
}

void DoseCalculator::set_offset(f32 ox, f32 oy, f32 oz) {
    dose.data_h.ox = ox;
    dose.data_h.oy = oy;
    dose.data_h.oz = oz;
}

/// Init
void DoseCalculator::initialize(GlobalSimulationParameters params)
{

    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("Dose calculator, size or spacing are set to zero?!");
        exit_simulation();
    }

    // Initi nb of voxels
    dose.data_h.nb_of_voxels = dose.data_h.nx*dose.data_h.ny*dose.data_h.nz;

    // CPU allocation
    m_cpu_malloc_dose();

    // Init values to 0 or 1
    for(int i = 0; i< dose.data_h.nb_of_voxels ; i++)
    {

        dose.data_h.edep[i] = 0.0;
        dose.data_h.dose[i] = 0.0;
        dose.data_h.edep_squared[i] = 0.0;
        dose.data_h.number_of_hits[i] = 0.0;
        dose.data_h.uncertainty[i] = 1.0;

    }

    // Copy to GPU if required
    if (params.data_h.device_target == GPU_DEVICE) {
        // GPU allocation
        m_gpu_malloc_dose();
        // Copy data to the GPU
        m_copy_dose_cpu2gpu();
    }

}


// __host__ __device__ void DoseCalculator::store_energy_and_energy2(ui32 voxel, f32 energy)
// {
// 
// #if defined(__CUDA_ARCH__)
//  ggems_atomic_add(dose_d.edep, voxel, energy);
//  ggems_atomic_add(dose_d.edep_squared, voxel, energy*energy);
// #else
//  ggems_atomic_add(dose_h.edep, voxel, energy);
//  ggems_atomic_add(dose_h.edep_squared, voxel, energy*energy);
// #endif
// 
// }



/*
void DoseCalculator::write_dosi(std::string histname)
{


ImageReader::record3Dimage(  histname,  
dose_h.edep,
make_f32xyz(dose_h.x0,dose_h.y0,dose_h.z0), 
make_f32xyz(dose_h.spacing_x,dose_h.spacing_y,dose_h.spacing_z),
make_i32xyz(dose_h.nx,dose_h.ny,dose_h.nz) ,
false);

}
*/

/// Private
bool DoseCalculator::m_check_mandatory() {
    if (dose.data_h.nx == 0 || dose.data_h.ny == 0 || dose.data_h.nz == 0 ||
        dose.data_h.spacing_x == 0 || dose.data_h.spacing_y == 0 || dose.data_h.spacing_z == 0) return false;
    else return true;
}

void DoseCalculator::m_cpu_malloc_dose() {
    dose.data_h.edep = new f32[dose.data_h.nb_of_voxels];
    dose.data_h.dose = new f32[dose.data_h.nb_of_voxels];
    dose.data_h.edep_squared = new f32[dose.data_h.nb_of_voxels];
    dose.data_h.number_of_hits = new ui32[dose.data_h.nb_of_voxels];
    dose.data_h.uncertainty = new f32[dose.data_h.nb_of_voxels];
}

void DoseCalculator::m_gpu_malloc_dose() {
    // GPU allocation
    HANDLE_ERROR( cudaMalloc((void**) &dose.data_d.edep,           dose.data_h.nb_of_voxels * sizeof(f32)));
    HANDLE_ERROR( cudaMalloc((void**) &dose.data_d.dose,           dose.data_h.nb_of_voxels * sizeof(f32)));
    HANDLE_ERROR( cudaMalloc((void**) &dose.data_d.edep_squared,   dose.data_h.nb_of_voxels * sizeof(f32)));
    HANDLE_ERROR( cudaMalloc((void**) &dose.data_d.number_of_hits, dose.data_h.nb_of_voxels * sizeof(ui32)));
    HANDLE_ERROR( cudaMalloc((void**) &dose.data_d.uncertainty,    dose.data_h.nb_of_voxels * sizeof(f32)));
}

void DoseCalculator::m_copy_dose_cpu2gpu()
{
    dose.data_d.nx = dose.data_h.nx;
    dose.data_d.ny = dose.data_h.ny;
    dose.data_d.nz = dose.data_h.nz;

    dose.data_d.spacing_x = dose.data_h.spacing_x;
    dose.data_d.spacing_y = dose.data_h.spacing_y;
    dose.data_d.spacing_z = dose.data_h.spacing_z;

    dose.data_d.ox = dose.data_h.ox;
    dose.data_d.oy = dose.data_h.oy;
    dose.data_d.oz = dose.data_h.oz;

    dose.data_d.nb_of_voxels = dose.data_h.nb_of_voxels;

    // Copy values to GPU arrays
    HANDLE_ERROR( cudaMemcpy(dose.data_d.edep,           dose.data_h.edep,           sizeof(f32)*dose.data_h.nb_of_voxels, cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dose.data_d.dose,           dose.data_h.dose,           sizeof(f32)*dose.data_h.nb_of_voxels, cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dose.data_d.edep_squared,   dose.data_h.edep_squared,   sizeof(f32)*dose.data_h.nb_of_voxels, cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dose.data_d.number_of_hits, dose.data_h.number_of_hits, sizeof(ui32)*dose.data_h.nb_of_voxels, cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dose.data_d.uncertainty,    dose.data_h.uncertainty,    sizeof(f32)*dose.data_h.nb_of_voxels, cudaMemcpyHostToDevice));
}

















#endif
