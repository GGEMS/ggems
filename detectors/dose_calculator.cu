// GGEMS Copyright (C) 2015

/*!
 * \file dose_calculator.cu
 * \brief
 * \author Y. Lemaréchal
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 02/12/2015
 * \date 26/02/2016, add volume of interest, change offset handling and fix many bugs - JB
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

    if (px < dose.xmin + EPSILON3 || px > dose.xmax - EPSILON3) return;
    if (py < dose.ymin + EPSILON3 || py > dose.ymax - EPSILON3) return;
    if (pz < dose.zmin + EPSILON3 || pz > dose.zmax - EPSILON3) return;

    // Defined index phantom    
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( px + dose.offset.x ) * dose.inv_doxel_size.x );
    index_phantom.y = ui32 ( ( py + dose.offset.y ) * dose.inv_doxel_size.y );
    index_phantom.z = ui32 ( ( pz + dose.offset.z ) * dose.inv_doxel_size.z );
    index_phantom.w = index_phantom.z * dose.slice_nb_doxels + index_phantom.y * dose.nb_doxels.x + index_phantom.x;

    //printf("Edep %e  pos %e %e %e  index %i\n", Edep, px, py, pz, index_phantom.w);

#ifdef DEBUG

    if ( index_phantom.x >= dose.nb_doxels.x || index_phantom.y >= dose.nb_doxels.y || index_phantom.z >= dose.nb_doxels.z)
    {
        printf(" IndexX %i  NbDox %i  px %f  Off %f invDox %f\n", index_phantom.x, dose.nb_doxels.x, px, dose.offset.x, dose.inv_doxel_size.x);
        printf(" IndexY %i  NbDox %i  py %f  Off %f invDox %f\n", index_phantom.y, dose.nb_doxels.y, py, dose.offset.y, dose.inv_doxel_size.y);
        printf(" IndexZ %i  NbDox %i  pz %f  Off %f invDox %f\n", index_phantom.z, dose.nb_doxels.z, pz, dose.offset.z, dose.inv_doxel_size.z);
        //index_phantom.z = 0;
    }

    assert( index_phantom.x < dose.nb_doxels.x );
    assert( index_phantom.y < dose.nb_doxels.y );
    assert( index_phantom.z < dose.nb_doxels.z );
#endif



    ggems_atomic_add_f64( dose.edep, index_phantom.w, Edep );
    ggems_atomic_add_f64( dose.edep_squared, index_phantom.w, Edep*Edep );
    ggems_atomic_add( dose.number_of_hits, index_phantom.w, ui32 ( 1 ) );

}

__host__ __device__ void dose_uncertainty_calculation ( DoseData dose, ui32 doxel_id_x, ui32 doxel_id_y, ui32 doxel_id_z )
{


    //              /                                    \ ^1/2
    //              |    N*Sum(Edep^2) - Sum(Edep)^2     |
    //  relError =  | __________________________________ |
    //              |                                    |
    //              \         (N-1)*Sum(Edep)^2          /
    //
    //   where Edep represents the energy deposit in one hit and N the number of energy deposits (hits)

    ui32 index = doxel_id_z * dose.slice_nb_doxels + doxel_id_y * dose.nb_doxels.x + doxel_id_x;

    if (dose.number_of_hits[index] > 1)
    {

        f64 sum2_E = dose.edep[index] * dose.edep[index];

        f64 num = ( dose.number_of_hits[index] * dose.edep_squared[index] ) - sum2_E;
        f64 den = ( dose.number_of_hits[index] - 1 ) * sum2_E;

        dose.uncertainty[index] = pow ( num/den, 0.5 ) * 100.0;
    }
    else
    {
        dose.uncertainty[index] = 100.0;
    }

}

__host__ __device__ void dose_to_water_calculation ( DoseData dose, ui32 doxel_id_x, ui32 doxel_id_y, ui32 doxel_id_z )
{

    f64 vox_vol = dose.doxel_size.x * dose.doxel_size.y * dose.doxel_size.z;
    f64 density = 1.0 * gram/cm3;
    ui32 index = doxel_id_z * dose.slice_nb_doxels + doxel_id_y * dose.nb_doxels.x + doxel_id_x;
    dose.dose[index] = ( 1.602E-10/vox_vol ) * dose.edep[index]/density;
    // Mev2Joule conversion (1.602E-13) / density scaling (10E-3) from g/mm3 to kg/mm3

}

__host__ __device__ void dose_to_phantom_calculation ( DoseData dose, VoxVolumeData phan,
                                                       MaterialsTable materials, f32 dose_min_density,
                                                       ui32 doxel_id_x, ui32 doxel_id_y, ui32 doxel_id_z )
{

    f64 vox_vol = dose.doxel_size.x * dose.doxel_size.y * dose.doxel_size.z;

    // Convert doxel_id into position
    f32 pos_x = ( doxel_id_x * dose.doxel_size.x ) - dose.offset.x;
    f32 pos_y = ( doxel_id_y * dose.doxel_size.y ) - dose.offset.y;
    f32 pos_z = ( doxel_id_z * dose.doxel_size.z ) - dose.offset.z;

    // Convert position into phantom voxel index
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / phan.spacing_x;
    ivoxsize.y = 1.0 / phan.spacing_y;
    ivoxsize.z = 1.0 / phan.spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( pos_x + phan.off_x ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( pos_y + phan.off_y ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pos_z + phan.off_z ) * ivoxsize.z );
    index_phantom.w = index_phantom.z*phan.nb_vox_x*phan.nb_vox_y
                         + index_phantom.y*phan.nb_vox_x
                         + index_phantom.x; // linear index

#ifdef DEBUG
    assert( index_phantom.x < dose.nb_doxels.x );
    assert( index_phantom.y < dose.nb_doxels.y );
    assert( index_phantom.z < dose.nb_doxels.z );
#endif

    // Get density for this voxel
    f64 density = materials.density[ phan.values[ index_phantom.w ] ]; // density given by the material id

    // Compute the dose
    ui32 index = doxel_id_z * dose.slice_nb_doxels + doxel_id_y * dose.nb_doxels.x + doxel_id_x;
    if ( density > dose_min_density )
    {
        dose.dose[index] = ( 1.602E-10/vox_vol ) * dose.edep[index]/density;
        // Mev2Joule conversion (1.602E-13) / density scaling (10E-3) from g/mm3 to kg/mm3
    }
    else
    {
        dose.dose[index] = 0.0f;
    }

}

/// Class
DoseCalculator::DoseCalculator()
{
    m_doxel_size.x = 0;
    m_doxel_size.y = 0;
    m_doxel_size.z = 0;

    m_offset.x = FLT_MAX;
    m_offset.y = FLT_MAX;
    m_offset.z = FLT_MAX;

    m_nb_of_doxels.x = 0;
    m_nb_of_doxels.y = 0;
    m_nb_of_doxels.z = 0;

    m_xmin = 0; m_xmax = 0;
    m_ymin = 0; m_ymax = 0;
    m_zmin = 0; m_zmax = 0;

    // Min density to compute the dose
    m_dose_min_density = 0.0;

    m_flag_phantom = false;
    m_flag_materials = false;
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

/*
void DoseCalculator::set_size_in_voxel ( ui32 nx, ui32 ny, ui32 nz )
{
    dose.data_h.nx = nx;
    dose.data_h.ny = ny;
    dose.data_h.nz = nz;
    dose.data_h.nb_of_voxels = nx*ny*nz;
}
*/

void DoseCalculator::set_doxel_size ( f32 sx, f32 sy, f32 sz )
{
    m_doxel_size = make_f32xyz( sx, sy, sz );
}

void DoseCalculator::set_voi( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )
{
    m_xmin = xmin; m_xmax = xmax;
    m_ymin = ymin; m_ymax = ymax;
    m_zmin = zmin; m_zmax = zmax;
}

/*
void DoseCalculator::set_offset ( f32 ox, f32 oy, f32 oz )
{
    dose.data_h.ox = ox;
    dose.data_h.oy = oy;
    dose.data_h.oz = oz;
}
*/

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

// In g/cm3 ? TODO - JB
void DoseCalculator::set_min_density ( f32 min )
{
    m_dose_min_density = min * gram/mm3;
}

/// Init
void DoseCalculator::initialize ( GlobalSimulationParameters params )
{
//     GGcout << " DoseCalculator initialize " << GGendl;
    
    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        print_error ( "Dose calculator, phantom and materials are not set?!" );
        exit_simulation();
    }

    // Copy params
    m_params = params;

    /// Compute dosemap parameters /////////////////////////////

    // Select a doxel size
    if ( m_doxel_size.x > 0.0 && m_doxel_size.y > 0.0 && m_doxel_size.z > 0.0 )
    {
        dose.data_h.doxel_size = m_doxel_size;
        dose.data_h.inv_doxel_size = fxyz_inv( m_doxel_size );
    }
    else
    {
        dose.data_h.doxel_size = make_f32xyz( m_phantom.data_h.spacing_x,
                                              m_phantom.data_h.spacing_y,
                                              m_phantom.data_h.spacing_z );
        dose.data_h.inv_doxel_size = fxyz_inv( dose.data_h.doxel_size );
    }

    // Compute min-max volume of interest
    f32xyz phan_size = make_f32xyz( m_phantom.data_h.nb_vox_x * m_phantom.data_h.spacing_x,
                                    m_phantom.data_h.nb_vox_y * m_phantom.data_h.spacing_y,
                                    m_phantom.data_h.nb_vox_z * m_phantom.data_h.spacing_z );
    f32xyz half_phan_size = fxyz_scale( phan_size, 0.5f );
    f32 phan_xmin = -half_phan_size.x; f32 phan_xmax = half_phan_size.x;
    f32 phan_ymin = -half_phan_size.y; f32 phan_ymax = half_phan_size.y;
    f32 phan_zmin = -half_phan_size.z; f32 phan_zmax = half_phan_size.z;

    // Select a min-max VOI
    if ( !m_xmin && !m_xmax && !m_ymin && !m_ymax && !m_zmin && !m_zmax )
    {
        dose.data_h.xmin = phan_xmin;
        dose.data_h.xmax = phan_xmax;
        dose.data_h.ymin = phan_ymin;
        dose.data_h.ymax = phan_ymax;
        dose.data_h.zmin = phan_zmin;
        dose.data_h.zmax = phan_zmax;
    }

    // Get the current dimension of the dose map
    f32xyz cur_dose_size = make_f32xyz( dose.data_h.xmax - dose.data_h.xmin,
                                        dose.data_h.ymax - dose.data_h.ymin,
                                        dose.data_h.zmax - dose.data_h.zmin );

    // New nb of voxels
    dose.data_h.nb_doxels.x = floor( cur_dose_size.x / dose.data_h.doxel_size.x );
    dose.data_h.nb_doxels.y = floor( cur_dose_size.y / dose.data_h.doxel_size.y );
    dose.data_h.nb_doxels.z = floor( cur_dose_size.z / dose.data_h.doxel_size.z );
    dose.data_h.slice_nb_doxels = dose.data_h.nb_doxels.x * dose.data_h.nb_doxels.y;
    dose.data_h.tot_nb_doxels = dose.data_h.slice_nb_doxels * dose.data_h.nb_doxels.z;

    // Compute the new size (due to integer nb of doxels)
    f32xyz new_dose_size = fxyz_mul( dose.data_h.doxel_size, cast_ui32xyz_to_f32xyz( dose.data_h.nb_doxels ) );

    if ( new_dose_size.x <= 0.0 || new_dose_size.y <= 0.0 || new_dose_size.z <= 0.0 )
    {
        GGcerr << "Dosemap dimension: "
               << new_dose_size.x << " "
               << new_dose_size.y << " "
               << new_dose_size.z << GGendl;
        exit_simulation();
    }

    // Compute new min and max after voxel alignment
    f32xyz half_delta_size = fxyz_scale( fxyz_sub( cur_dose_size, new_dose_size ), 0.5f );

    dose.data_h.xmin += half_delta_size.x;
    dose.data_h.xmax -= half_delta_size.x;

    dose.data_h.ymin += half_delta_size.y;
    dose.data_h.ymax -= half_delta_size.y;

    dose.data_h.zmin += half_delta_size.z;
    dose.data_h.zmax -= half_delta_size.z;

    // Get the new offset
    dose.data_h.offset.x = m_phantom.data_h.off_x - ( dose.data_h.xmin - phan_xmin );
    dose.data_h.offset.y = m_phantom.data_h.off_y - ( dose.data_h.ymin - phan_ymin );
    dose.data_h.offset.z = m_phantom.data_h.off_z - ( dose.data_h.zmin - phan_zmin );

//    printf("Nb doxels %i %i %i   Dose size %f %f %f\n", dose.data_h.nb_doxels.x, dose.data_h.nb_doxels.y, dose.data_h.nb_doxels.z,
//                                                        new_dose_size.x, new_dose_size.y, new_dose_size.z);
//    printf("Offset %f %f %f\n", dose.data_h.offset.x, dose.data_h.offset.y, dose.data_h.offset.z);


    //////////////////////////////////////////////////////////

    // CPU allocation
    m_cpu_malloc_dose();

    // Init values to 0 or 1
    for ( int i = 0; i< dose.data_h.tot_nb_doxels ; i++ )
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
    for ( ui32 iz=0; iz < dose.data_h.nb_doxels.z; iz++ )
    {
        for ( ui32 iy=0; iy < dose.data_h.nb_doxels.y; iy++ )
        {
            for ( ui32 ix=0; ix < dose.data_h.nb_doxels.x; ix++ )
            {
                dose_to_water_calculation ( dose.data_h, ix, iy, iz );
                dose_uncertainty_calculation ( dose.data_h, ix, iy, iz );
            }
        }
    }

    m_flag_dose_calculated = true;
}

void DoseCalculator::calculate_dose_to_phantom()
{
    // Check if everything was set properly
    if ( !m_flag_materials || !m_flag_phantom )
    {
        GGcerr << "Dose calculator, phantom and materials data are required!" << GGendl;
        exit_simulation();
    }

    // First get back data if stored on GPU
    if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        m_copy_dose_gpu2cpu();
    }

    // Calculate the dose to phantom and the uncertainty
    for ( ui32 iz=0; iz < dose.data_h.nb_doxels.z; iz++ )
    {
        for ( ui32 iy=0; iy < dose.data_h.nb_doxels.y; iy++ )
        {
            for ( ui32 ix=0; ix < dose.data_h.nb_doxels.x; ix++ )
            {
                dose_to_phantom_calculation ( dose.data_h, m_phantom.data_h, m_materials.data_h,
                                              m_dose_min_density, ix, iy, iz );
                dose_uncertainty_calculation ( dose.data_h, ix, iy, iz );
            }
        }
    }
    
    m_flag_dose_calculated = true;
}

/// Private
bool DoseCalculator::m_check_mandatory()
{
    if ( !m_flag_materials || !m_flag_phantom ) return false;
    else return true;
}

void DoseCalculator::m_cpu_malloc_dose()
{
    dose.data_h.edep = new f64[dose.data_h.tot_nb_doxels];
    dose.data_h.dose = new f64[dose.data_h.tot_nb_doxels];
    dose.data_h.edep_squared = new f64[dose.data_h.tot_nb_doxels];
    dose.data_h.number_of_hits = new ui32[dose.data_h.tot_nb_doxels];
    dose.data_h.uncertainty = new f64[dose.data_h.tot_nb_doxels];
}

void DoseCalculator::m_gpu_malloc_dose()
{
    // GPU allocation
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.edep,           dose.data_h.tot_nb_doxels * sizeof ( f64 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.dose,           dose.data_h.tot_nb_doxels * sizeof ( f64 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.edep_squared,   dose.data_h.tot_nb_doxels * sizeof ( f64 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.number_of_hits, dose.data_h.tot_nb_doxels * sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dose.data_d.uncertainty,    dose.data_h.tot_nb_doxels * sizeof ( f64 ) ) );
    
//     GGcout << "DoseCalculator : GPU allocation " << dose.data_h.nb_of_voxels << GGendl;
    
}

void DoseCalculator::m_copy_dose_cpu2gpu()
{

    dose.data_d.nb_doxels = dose.data_h.nb_doxels;
    dose.data_d.doxel_size = dose.data_h.doxel_size;
    dose.data_d.inv_doxel_size = dose.data_h.inv_doxel_size;
    dose.data_d.offset = dose.data_h.offset;
    dose.data_d.tot_nb_doxels = dose.data_h.tot_nb_doxels;
    dose.data_d.slice_nb_doxels = dose.data_h.slice_nb_doxels;

    dose.data_d.xmin = dose.data_h.xmin;
    dose.data_d.xmax = dose.data_h.xmax;
    dose.data_d.ymin = dose.data_h.ymin;
    dose.data_d.ymax = dose.data_h.ymax;
    dose.data_d.zmin = dose.data_h.zmin;
    dose.data_d.zmax = dose.data_h.zmax;

    // Copy values to GPU arrays
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.edep,           dose.data_h.edep,           sizeof ( f64 ) *dose.data_h.tot_nb_doxels,  cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.dose,           dose.data_h.dose,           sizeof ( f64 ) *dose.data_h.tot_nb_doxels,  cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.edep_squared,   dose.data_h.edep_squared,   sizeof ( f64 ) *dose.data_h.tot_nb_doxels,  cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.number_of_hits, dose.data_h.number_of_hits, sizeof ( ui32 ) *dose.data_h.tot_nb_doxels, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_d.uncertainty,    dose.data_h.uncertainty,    sizeof ( f64 ) *dose.data_h.tot_nb_doxels,  cudaMemcpyHostToDevice ) );
    
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

//     dose.data_h.nb_of_voxels = dose.data_d.nb_of_voxels;

//     GGcout << "DoseCalculator : Copy to GPU " << dose.data_h.nb_of_voxels << GGendl;
    // Copy values to GPU arrays
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.edep,           dose.data_d.edep,           sizeof ( f64  ) *dose.data_h.tot_nb_doxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.dose,           dose.data_d.dose,           sizeof ( f64  ) *dose.data_h.tot_nb_doxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.edep_squared,   dose.data_d.edep_squared,   sizeof ( f64  ) *dose.data_h.tot_nb_doxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.number_of_hits, dose.data_d.number_of_hits, sizeof ( ui32 ) *dose.data_h.tot_nb_doxels,  cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( dose.data_h.uncertainty,    dose.data_d.uncertainty,    sizeof ( f64  ) *dose.data_h.tot_nb_doxels,  cudaMemcpyDeviceToHost ) );

}



void DoseCalculator::write ( std::string filename )
{

    
    if ( ( m_params.data_h.device_target == GPU_DEVICE ) && (!m_flag_dose_calculated) )
    {
        m_copy_dose_gpu2cpu();
    }
    
    std::string format = ImageReader::get_format ( filename );
    filename = ImageReader::get_filename_without_format ( filename );
            
    ImageReader::record3Dimage (
        filename + "-Edep."+format ,
        dose.data_h.edep, dose.data_h.offset, dose.data_h.doxel_size, dose.data_h.nb_doxels );

    if ( m_flag_dose_calculated )
    {

        ImageReader::record3Dimage (
                    filename + "-Dose."+format,
                    dose.data_h.dose,dose.data_h.offset, dose.data_h.doxel_size, dose.data_h.nb_doxels );

        ImageReader::record3Dimage (
                    filename + "-Uncertainty."+format,
                    dose.data_h.uncertainty, dose.data_h.offset, dose.data_h.doxel_size, dose.data_h.nb_doxels );


        ImageReader::record3Dimage (
                    filename + "-Hit."+format,
                    dose.data_h.number_of_hits, dose.data_h.offset, dose.data_h.doxel_size, dose.data_h.nb_doxels );
                
    }
    else
    {
        GGcout << "Dose calculation was not request, there is no dose map to export" << GGendl;
    }


}



#endif
