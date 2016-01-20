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
__host__ __device__ void dose_record_standard ( DoseData &dose, f32 Edep, f32 px, f32 py, f32 pz )
{

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / dose.spacing_x;
    ivoxsize.y = 1.0 / dose.spacing_y;
    ivoxsize.z = 1.0 / dose.spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( px-dose.ox ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( py-dose.oy ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pz-dose.oz ) * ivoxsize.z );
    index_phantom.w = index_phantom.z*dose.nx*dose.ny
                      + index_phantom.y*dose.nx
                      + index_phantom.x; // linear index

    if ( ( index_phantom.x >= dose.nx ) ||
            ( index_phantom.y >= dose.ny ) ||
            ( index_phantom.z >= dose.nz ) )
    {

// GGcout <<  index_phantom.x << "  " <<  index_phantom.y << "  " <<  index_phantom.z << GGendl;
// GGcout << pz << "   " << dose.oz << "   " << ivoxsize.z << GGendl;
// GGcout <<  dose.nx << "  " <<  dose.ny << "  " <<  dose.nz << GGendl;
// GGcout <<  px << "  " <<  py << "  " <<  pz << GGendl;
printf("Error out of dosemap \n");
        return;
    }
    
//     if((index_phantom.x < 20 ) &&
//       (index_phantom.y < 20 ) &&
//       (index_phantom.z < 20 ))
//       printf("Edep : %g, Index : %d %d %d \n",Edep, index_phantom.x,index_phantom.y,index_phantom.z);
    
// GGcout <<  px << "  " <<  py << "  " <<  pz << GGendl;
// GGcout <<  index_phantom.x << "  " <<  index_phantom.y << "  " <<  index_phantom.z << GGendl;

// GGcout <<  ivoxsize.x << "  " <<  ivoxsize.y << "  " <<  ivoxsize.z << GGendl;
// GGcout <<  dose.spacing_x << "  " <<  dose.spacing_y << "  " <<  dose.spacing_z << GGendl;
// GGcout << dose.nb_of_voxels << GGendl;
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
// //     Score dosemap
// GGcout  << "   " << index_phantom.w << "   " << Edep << GGendl;
//
if(Edep<0.)
{
//     GGcout << "WTF Dose < 0? " << Edep << GGendl; 
    return;
}

    ggems_atomic_add ( dose.edep, index_phantom.w, Edep );
    ggems_atomic_add ( dose.edep_squared, index_phantom.w, Edep*Edep );
    ggems_atomic_add ( dose.number_of_hits, index_phantom.w, ui32 ( 1 ) );
//  GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
}

__host__ __device__ void dose_uncertainty_calculation ( DoseData dose, ui32 id )
{


    //              /                                    \ ^1/2
    //              |    N*Sum(Edep^2) - Sum(Edep)^2     |
    //  relError =  | __________________________________ |
    //              |                                    |
    //              \         (N-1)*Sum(Edep)^2          /
    //
    //   where Edep represents the energy deposit in one hit and N the number of energy deposits (hits)

    f64 sum2_E = dose.edep[id] * dose.edep[id];

    f64 num = ( dose.number_of_hits[id] * dose.edep_squared[id] ) - sum2_E;
    f64 den = ( dose.number_of_hits[id]-1 ) * sum2_E;

    dose.uncertainty[id] = pow ( num/den, 0.5 );

}

__host__ __device__ void dose_to_water_calculation ( DoseData dose, ui32 id )
{

    f64 vox_vol = dose.spacing_x*dose.spacing_y*dose.spacing_z;
    f64 density = 1.0 * gram/cm3;
    dose.dose[id] = ( 1.602E-10/vox_vol ) * dose.edep[id]/density; // Mev2Joule conversion (1.602E-13) / density scaling (10E-3) from g/mm3 to kg/mm3

}

__host__ __device__ void dose_to_phantom_calculation ( DoseData dose, VoxVolumeData volume, MaterialsTable materials, f32 dose_min_density, ui32 id )
{

    f64 vox_vol = dose.spacing_x*dose.spacing_y*dose.spacing_z;
    f64 density = materials.density[volume.values[id]]; // density given by the material id
    if ( density > dose_min_density )
    {
        dose.dose[id] = ( 1.602E-10/vox_vol ) * dose.edep[id]/density; // Mev2Joule conversion (1.602E-13) / density scaling (10E-3) from g/mm3 to kg/mm3
    }
    else
    {
        dose.dose[id] = 0.0f;
    }

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

    // No phan tom assigned yet
    m_flag_phantom = false;
    m_flag_materials = false;

    // Min density to compute the dose
    m_dose_min_density = 0.0;
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
void DoseCalculator::set_size_in_voxel ( ui32 nx, ui32 ny, ui32 nz )
{
    dose.data_h.nx = nx;
    dose.data_h.ny = ny;
    dose.data_h.nz = nz;
}

void DoseCalculator::set_voxel_size ( f32 sx, f32 sy, f32 sz )
{
    dose.data_h.spacing_x = sx;
    dose.data_h.spacing_y = sy;
    dose.data_h.spacing_z = sz;
}

void DoseCalculator::set_offset ( f32 ox, f32 oy, f32 oz )
{
    dose.data_h.ox = ox;
    dose.data_h.oy = oy;
    dose.data_h.oz = oz;
}

void DoseCalculator::set_voxelized_phantom ( VoxelizedPhantom aphantom )
{
    m_phantom = aphantom;
    m_flag_phantom = true;
}

void DoseCalculator::set_materials ( Materials materials )
{
    m_materials = materials;
    m_flag_materials = true;
}

// In g/cm3 ?
void DoseCalculator::set_min_density ( f32 min )
{
    m_dose_min_density = min * gram/mm3;
}

/// Init
void DoseCalculator::initialize ( GlobalSimulationParameters params )
{
    GGcout << " DoseCalculator initialize " << GGendl;
    
    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        print_error ( "Dose calculator, size or spacing are set to zero?!" );
        exit_simulation();
    }

    // Initi nb of voxels
    dose.data_h.nb_of_voxels = dose.data_h.nx*dose.data_h.ny*dose.data_h.nz;

    // Copy params
    m_params = params;

    // CPU allocation
    m_cpu_malloc_dose();

    // Init values to 0 or 1
    for ( int i = 0; i< dose.data_h.nb_of_voxels ; i++ )
    {

        dose.data_h.edep[i] = 0.0;
        dose.data_h.dose[i] = 0.0;
        dose.data_h.edep_squared[i] = 0.0;
        dose.data_h.number_of_hits[i] = 0.0;
        dose.data_h.uncertainty[i] = 1.0;

    }

    // Copy to GPU if required
    if ( params.data_h.device_target == GPU_DEVICE )
    {
        // GPU allocation
        m_gpu_malloc_dose();
        // Copy data to the GPU
        m_copy_dose_cpu2gpu();
    }

}

void DoseCalculator::calculate_dose_to_water()
{

    // First get back data if stored on GPU
    if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        m_copy_dose_gpu2cpu();
    }

    // Calculate the dose to water and the uncertainty
    ui32 id=0;
    while ( id<dose.data_h.nb_of_voxels )
    {
        dose_to_water_calculation ( dose.data_h, id );
        dose_uncertainty_calculation ( dose.data_h, id );
        ++id;
    }
}

void DoseCalculator::calculate_dose_to_phantom()
{

    // Check if everything was set properly
    if ( !m_flag_materials || !m_flag_phantom )
    {
        print_error ( "Dose calculator, phantom and materials data are required!" );
        exit_simulation();
    }

    // First get back data if stored on GPU
    if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        m_copy_dose_gpu2cpu();
    }

    // Calculate the dose to phantom and the uncertainty
    ui32 id=0;
    while ( id<dose.data_h.nb_of_voxels )
    {
        dose_to_phantom_calculation ( dose.data_h, m_phantom.data_h, m_materials.data_h, m_dose_min_density, id );
        dose_uncertainty_calculation ( dose.data_h, id );
        ++id;
    }
}

/// Private
bool DoseCalculator::m_check_mandatory()
{
    if ( dose.data_h.nx == 0 || dose.data_h.ny == 0 || dose.data_h.nz == 0 ||
            dose.data_h.spacing_x == 0 || dose.data_h.spacing_y == 0 || dose.data_h.spacing_z == 0 ) return false;
    else return true;
}

void DoseCalculator::m_cpu_malloc_dose()
{
    dose.data_h.edep = new f64[dose.data_h.nb_of_voxels];
    dose.data_h.dose = new f64[dose.data_h.nb_of_voxels];
    dose.data_h.edep_squared = new f64[dose.data_h.nb_of_voxels];
    dose.data_h.number_of_hits = new ui32[dose.data_h.nb_of_voxels];
    dose.data_h.uncertainty = new f64[dose.data_h.nb_of_voxels];
}

void DoseCalculator::m_gpu_malloc_dose()
{
    // GPU allocation
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.edep,           dose.data_h.nb_of_voxels * sizeof ( f64 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.dose,           dose.data_h.nb_of_voxels * sizeof ( f64 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.edep_squared,   dose.data_h.nb_of_voxels * sizeof ( f64 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.number_of_hits, dose.data_h.nb_of_voxels * sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.uncertainty,    dose.data_h.nb_of_voxels * sizeof ( f64 ) ) );
    
    GGcout << " DoseCalculator GPU allocation " << GGendl;
    
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
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.edep,           dose.data_h.edep,           sizeof ( f32 ) *dose.data_h.nb_of_voxels,  cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.dose,           dose.data_h.dose,           sizeof ( f32 ) *dose.data_h.nb_of_voxels,  cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.edep_squared,   dose.data_h.edep_squared,   sizeof ( f32 ) *dose.data_h.nb_of_voxels,  cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.number_of_hits, dose.data_h.number_of_hits, sizeof ( ui32 ) *dose.data_h.nb_of_voxels, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.uncertainty,    dose.data_h.uncertainty,    sizeof ( f32 ) *dose.data_h.nb_of_voxels,  cudaMemcpyHostToDevice ) );
    
     GGcout << " Copy dose calculator to GPU " << GGendl;
    
}

void DoseCalculator::m_copy_dose_gpu2cpu()
{
//     dose.data_h.nx = dose.data_d.nx;
//     dose.data_h.ny = dose.data_d.ny;
//     dose.data_h.nz = dose.data_d.nz;
// 
//     dose.data_h.spacing_x = dose.data_d.spacing_x;
//     dose.data_h.spacing_y = dose.data_d.spacing_y;
//     dose.data_h.spacing_z = dose.data_d.spacing_z;
// 
//     dose.data_h.ox = dose.data_d.ox;
//     dose.data_h.oy = dose.data_d.oy;
//     dose.data_h.oz = dose.data_d.oz;
// 
//     dose.data_h.nb_of_voxels = dose.data_d.nb_of_voxels;

    // Copy values to GPU arrays
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.edep,           dose.data_d.edep,           sizeof ( f32  ) *dose.data_h.nb_of_voxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.dose,           dose.data_d.dose,           sizeof ( f32  ) *dose.data_h.nb_of_voxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.edep_squared,   dose.data_d.edep_squared,   sizeof ( f32  ) *dose.data_h.nb_of_voxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.number_of_hits, dose.data_d.number_of_hits, sizeof ( ui32 ) *dose.data_h.nb_of_voxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.uncertainty,    dose.data_d.uncertainty,    sizeof ( f32  ) *dose.data_h.nb_of_voxels,  cudaMemcpyDeviceToHost ) );
    
    GGcout << " Copy dose calculator to CPU " << GGendl;
    GGcout << " GPU size : " << dose.data_h.nb_of_voxels << GGendl;
}



void DoseCalculator::write ( std::string filename )
{

    GGcout << "Write image " << filename << GGendl;
    GGcout << "Size : " << dose.data_h.nx * dose.data_h.ny * dose.data_h.nz << GGendl;
    if ( m_params.data_h.device_target == GPU_DEVICE )
        m_copy_dose_gpu2cpu();

    ImageReader::record3Dimage (
        filename,
        dose.data_h.edep,
        make_f32xyz ( dose.data_h.ox,dose.data_h.oy,dose.data_h.oz ),
        make_f32xyz ( dose.data_h.spacing_x,dose.data_h.spacing_y,dose.data_h.spacing_z ),
        make_i32xyz ( dose.data_h.nx,dose.data_h.ny,dose.data_h.nz ) );

}



#endif
