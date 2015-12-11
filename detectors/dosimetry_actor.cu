#ifndef DOSIMETRY_MANAGER_CU
#define DOSIMETRY_MANAGER_CU

#include "dosimetry_actor.cuh"


DoseCalculator::DoseCalculator()
{

    dose_h.nx = 0;
    dose_h.ny = 0;
    dose_h.nz = 0;
    
    // Voxel size per dimension
    dose_h.spacing_x = 0.0;
    dose_h.spacing_y = 0.0;
    dose_h.spacing_z = 0.0;
    
    // Offset
    dose_h.x0 = 0.0;
    dose_h.y0 = 0.0;
    dose_h.z0 = 0.0;
    
    ui32 nb_of_voxels = 0;

        
    dose_h.edep = NULL;
    dose_h.dose = NULL;
    dose_h.edep_squared = NULL;
    dose_h.number_of_hits = NULL;
    
    dose_h.uncertainty = NULL;
}

DoseCalculator::~DoseCalculator()
{

    delete [] dose_h.edep ;
    delete [] dose_h.dose ;
    delete [] dose_h.edep_squared ;
    delete [] dose_h.number_of_hits ;
    delete [] dose_h.uncertainty ;
    

}


void DoseCalculator::initialize(ui32xyz nvox ,f32xyz spacing , f32xyz offset)
{

    dose_h.nb_of_voxels = ui32xyz_mul(nvox);

    dose_h.edep = new f32[dose_h.nb_of_voxels];
    dose_h.dose = new f32[dose_h.nb_of_voxels];
    dose_h.edep_squared = new f32[dose_h.nb_of_voxels];
    dose_h.number_of_hits = new ui32[dose_h.nb_of_voxels];
    
    dose_h.uncertainty = new f32[dose_h.nb_of_voxels];
    
    dose_h.nx = nvox.x;
    dose_h.ny = nvox.y;
    dose_h.nz = nvox.z;
    
    
    dose_h.spacing_x = spacing.x;
    dose_h.spacing_y = spacing.y;
    dose_h.spacing_z = spacing.z;
    
    dose_h.x0 = offset.x;
    dose_h.y0 = offset.y;
    dose_h.z0 = offset.z;
    
    for(int i = 0; i< dose_h.nb_of_voxels ; i++)
    {
    
        dose_h.edep[i] = 0.0;
        dose_h.dose[i] = 0.0;
        dose_h.edep_squared[i] = 0.0;
        dose_h.number_of_hits[i] = 0.0;
    
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


void DoseCalculator::m_copy_dosi_cpu2gpu()
{

    dose_d.nx = dose_h.nx;
    dose_d.ny = dose_h.ny;
    dose_d.nz = dose_h.nz;

    dose_d.spacing_x = dose_h.spacing_x;
    dose_d.spacing_y = dose_h.spacing_y;
    dose_d.spacing_z = dose_h.spacing_z;

    dose_d.x0 = dose_h.x0;
    dose_d.y0 = dose_h.y0;
    dose_d.z0 = dose_h.z0;

    ui32 nvox = dose_h.nb_of_voxels;

    
    // GPU allocation
    HANDLE_ERROR( cudaMalloc((void**) &dose_d.edep,         nvox * sizeof(f32)));
    HANDLE_ERROR( cudaMalloc((void**) &dose_d.dose,         nvox * sizeof(f32)));
    HANDLE_ERROR( cudaMalloc((void**) &dose_d.edep_squared, nvox * sizeof(f32)));

    HANDLE_ERROR( cudaMalloc((void**) &dose_d.number_of_hits, nvox * sizeof(ui32)));

    // Copy 0 values to GPU arrays
    HANDLE_ERROR( cudaMemcpy(dose_d.edep,           dose_h.edep,            sizeof(f32)*nvox, cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dose_d.dose,           dose_h.dose,            sizeof(f32)*nvox, cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dose_d.edep_squared,   dose_h.edep_squared,    sizeof(f32)*nvox, cudaMemcpyHostToDevice));

    HANDLE_ERROR( cudaMemcpy(dose_d.number_of_hits, dose_h.number_of_hits,  sizeof(ui32)*nvox, cudaMemcpyHostToDevice));
}



void DoseCalculator::write_dosi(std::string histname)
{


ImageReader::record3Dimage(  histname,  
dose_h.edep,
make_f32xyz(dose_h.x0,dose_h.y0,dose_h.z0), 
make_f32xyz(dose_h.spacing_x,dose_h.spacing_y,dose_h.spacing_z),
make_i32xyz(dose_h.nx,dose_h.ny,dose_h.nz) ,
false);

}


// std::ostream& operator<<(std::ostream& os, const DoseCalculator& v)
// {
//     os << v.dose_h.x0 ;
// //             os << v.x << '/' << v.y << '/' << v.z;
//     return os;
// }

#endif