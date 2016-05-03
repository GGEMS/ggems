// GGEMS Copyright (C) 2015

/*!
 * \file dose_calculator.cu
 * \brief
 * \author Y. Lemaréchal
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 02/12/2015
 * \date 26/02/2016, add volume of interest, change offset handling and fix many bugs - JB
 * \date 18/04/2016, change every things, use unified memory, improve code - JB
 *
 *
 */

#ifndef DOSE_CALCULATOR_CU
#define DOSE_CALCULATOR_CU

#include "dose_calculator.cuh"

/// CPU&GPU functions //////////////////////////////////////////////////////////

// Analog deposition
__host__ __device__ void dose_record_standard ( DoseData &dose, f32 Edep, f32 px, f32 py, f32 pz )
{

    if (px < dose.xmin + EPSILON3 || px > dose.xmax - EPSILON3) return;
    if (py < dose.ymin + EPSILON3 || py > dose.ymax - EPSILON3) return;
    if (pz < dose.zmin + EPSILON3 || pz > dose.zmax - EPSILON3) return;

    // Defined index phantom    
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( px + dose.offset.x ) * dose.inv_dosel_size.x );
    index_phantom.y = ui32 ( ( py + dose.offset.y ) * dose.inv_dosel_size.y );
    index_phantom.z = ui32 ( ( pz + dose.offset.z ) * dose.inv_dosel_size.z );
    index_phantom.w = index_phantom.z * dose.slice_nb_dosels + index_phantom.y * dose.nb_dosels.x + index_phantom.x;

    //printf("Edep %e  pos %e %e %e  index %i\n", Edep, px, py, pz, index_phantom.w);

#ifdef DEBUG

    if ( index_phantom.x >= dose.nb_dosels.x || index_phantom.y >= dose.nb_dosels.y || index_phantom.z >= dose.nb_dosels.z)
    {
        printf(" IndexX %i  NbDox %i  px %f  Off %f invDox %f\n", index_phantom.x, dose.nb_dosels.x, px, dose.offset.x, dose.inv_dosel_size.x);
        printf(" IndexY %i  NbDox %i  py %f  Off %f invDox %f\n", index_phantom.y, dose.nb_dosels.y, py, dose.offset.y, dose.inv_dosel_size.y);
        printf(" IndexZ %i  NbDox %i  pz %f  Off %f invDox %f\n", index_phantom.z, dose.nb_dosels.z, pz, dose.offset.z, dose.inv_dosel_size.z);
        //index_phantom.z = 0;
    }

    assert( index_phantom.x < dose.nb_dosels.x );
    assert( index_phantom.y < dose.nb_dosels.y );
    assert( index_phantom.z < dose.nb_dosels.z );
#endif

/*
#ifdef __CUDA_ARCH__
    atomicAdd(&dose.edep[index_phantom.w], Edep);
    atomicAdd(&dose.edep_squared[index_phantom.w], Edep*Edep);
    atomicAdd(&dose.number_of_hits[index_phantom.w], ui32(1));
#else
    dose.edep[index_phantom.w] += Edep;
    dose.edep_squared[index_phantom.w] += (Edep*Edep);
    dose.number_of_hits[index_phantom.w] += 1;
#endif
*/

    ggems_atomic_add_f64( dose.edep, index_phantom.w, f64( Edep ) );
    ggems_atomic_add_f64( dose.edep_squared, index_phantom.w, f64( Edep) * f64( Edep ) );
    ggems_atomic_add( dose.number_of_hits, index_phantom.w, ui32 ( 1 ) );                  // ui32, limited to 4.29e9 - JB


}

// TLE deposition
__host__ __device__ void dose_record_TLE ( DoseData &dose, f32 Edep, f32 px, f32 py, f32 pz,
                                           f32 length, f32 mu_en)
{

    if (px < dose.xmin + EPSILON3 || px > dose.xmax - EPSILON3) return;
    if (py < dose.ymin + EPSILON3 || py > dose.ymax - EPSILON3) return;
    if (pz < dose.zmin + EPSILON3 || pz > dose.zmax - EPSILON3) return;

    // Defined index phantom
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( px + dose.offset.x ) * dose.inv_dosel_size.x );
    index_phantom.y = ui32 ( ( py + dose.offset.y ) * dose.inv_dosel_size.y );
    index_phantom.z = ui32 ( ( pz + dose.offset.z ) * dose.inv_dosel_size.z );
    index_phantom.w = index_phantom.z * dose.slice_nb_dosels + index_phantom.y * dose.nb_dosels.x + index_phantom.x;

#ifdef DEBUG

    if ( index_phantom.x >= dose.nb_dosels.x || index_phantom.y >= dose.nb_dosels.y || index_phantom.z >= dose.nb_dosels.z)
    {
        printf(" IndexX %i  NbDox %i  px %f  Off %f invDox %f\n", index_phantom.x, dose.nb_dosels.x, px, dose.offset.x, dose.inv_dosel_size.x);
        printf(" IndexY %i  NbDox %i  py %f  Off %f invDox %f\n", index_phantom.y, dose.nb_dosels.y, py, dose.offset.y, dose.inv_dosel_size.y);
        printf(" IndexZ %i  NbDox %i  pz %f  Off %f invDox %f\n", index_phantom.z, dose.nb_dosels.z, pz, dose.offset.z, dose.inv_dosel_size.z);
        //index_phantom.z = 0;
    }

    assert( index_phantom.x < dose.nb_dosels.x );
    assert( index_phantom.y < dose.nb_dosels.y );
    assert( index_phantom.z < dose.nb_dosels.z );
#endif

    // TLE
    f64 energy_dropped = Edep * mu_en * length * 0.1; // arbitrary factor (see in GATE)

/*
#ifdef __CUDA_ARCH__
    atomicAdd(&dose.edep[index_phantom.w], energy_dropped);
    atomicAdd(&dose.edep_squared[index_phantom.w], energy_dropped*energy_dropped);
    atomicAdd(&dose.number_of_hits[index_phantom.w], ui32(1));
#else
    dose.edep[index_phantom.w] += energy_dropped;
    dose.edep_squared[index_phantom.w] += (energy_dropped*energy_dropped);
    dose.number_of_hits[index_phantom.w] += 1;
#endif
*/

    ggems_atomic_add_f64( dose.edep, index_phantom.w, energy_dropped );
    ggems_atomic_add_f64( dose.edep_squared, index_phantom.w, energy_dropped * energy_dropped );
    ggems_atomic_add( dose.number_of_hits, index_phantom.w, ui32 ( 1 ) );                  // ui32, limited to 4.29e9 - JB

}

/// Private /////////////////////////////////////////////////////////////////////

bool DoseCalculator::m_check_mandatory()
{
    if ( !m_flag_materials || !m_flag_phantom ) return false;
    else return true;
}

void DoseCalculator::m_uncertainty_calculation( ui32 dosel_id_x, ui32 dosel_id_y, ui32 dosel_id_z )
{

    // Relative statistical uncertainty (from Ma et al. PMB 47 2002 p1671) - JB
    //              /                                    \ ^1/2
    //              |    N*Sum(Edep^2) - Sum(Edep)^2     |
    //  relError =  | __________________________________ |
    //              |                                    |
    //              \         (N-1)*Sum(Edep)^2          /
    //
    //   where Edep represents the energy deposit in one hit and N the number of energy deposits (hits)

    // The same without developing - JB (from Walters, Kawrakow and Rogers Med. Phys. 29 2002)
    //                  /                                   \
    //             1    | Sum(Edep^2)      / Sum(Edep) \^2  |
    //  var(x) = _____  | ___________  --  |___________|    |
    //                  |                  |           |    |
    //            N-1   \     N            \    N      /    /
    //
    //  s(x) = ( var )^1/2
    //
    //  relError = s(x) / Sum(Edep)/N
    //

    ui32 index = dosel_id_z * dose.slice_nb_dosels + dosel_id_y * dose.nb_dosels.x + dosel_id_x;

    f64 N = dose.number_of_hits[index];
    f64 sum_E = dose.edep[index];

    if ( N > 1 && sum_E != 0.0 )
    {
        f64 sum_E2 = dose.edep_squared[index];
        f64 sum2_E = sum_E * sum_E;
        f64 s = ( (N*sum_E2) - sum2_E ) / ( (N-1) * sum2_E );

#ifdef DEBUG
        //assert(s >= 0.0);
        if ( s < 0.0 ) s = 1.0;
#endif
        m_uncertainty_values[ index ] = powf( s, 0.5 );
    }
    else
    {
        m_uncertainty_values[ index ] = 1.0;
    }

}

void DoseCalculator::m_dose_to_water_calculation( ui32 dosel_id_x, ui32 dosel_id_y, ui32 dosel_id_z )
{

    f64 vox_vol = dose.dosel_size.x * dose.dosel_size.y * dose.dosel_size.z;
    f64 density = 1.0 * gram/cm3;
    ui32 index = dosel_id_z * dose.slice_nb_dosels + dosel_id_y * dose.nb_dosels.x + dosel_id_x;

    m_dose_values[ index ] = dose.edep[ index ] / density / vox_vol / gray;

}

void DoseCalculator::m_dose_to_phantom_calculation( ui32 dosel_id_x, ui32 dosel_id_y, ui32 dosel_id_z )
{

    f64 vox_vol = dose.dosel_size.x * dose.dosel_size.y * dose.dosel_size.z;

    // Convert doxel_id into position
    f32 pos_x = ( dosel_id_x * dose.dosel_size.x ) - dose.offset.x;
    f32 pos_y = ( dosel_id_y * dose.dosel_size.y ) - dose.offset.y;
    f32 pos_z = ( dosel_id_z * dose.dosel_size.z ) - dose.offset.z;

    // Convert position into phantom voxel index
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / m_phantom.data_h.spacing_x;
    ivoxsize.y = 1.0 / m_phantom.data_h.spacing_y;
    ivoxsize.z = 1.0 / m_phantom.data_h.spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( pos_x + m_phantom.data_h.off_x ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( pos_y + m_phantom.data_h.off_y ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pos_z + m_phantom.data_h.off_z ) * ivoxsize.z );
    index_phantom.w = index_phantom.z*m_phantom.data_h.nb_vox_x*m_phantom.data_h.nb_vox_y
                         + index_phantom.y*m_phantom.data_h.nb_vox_x
                         + index_phantom.x; // linear index

#ifdef DEBUG
    assert( index_phantom.x < m_phantom.data_h.nb_vox_x );
    assert( index_phantom.y < m_phantom.data_h.nb_vox_y );
    assert( index_phantom.z < m_phantom.data_h.nb_vox_z );
#endif

    // Get density for this voxel
    f64 density = m_materials.data_h.density[ m_phantom.data_h.values[ index_phantom.w ] ]; // density given by the material id

    // Compute the dose
    ui32 index = dosel_id_z * dose.slice_nb_dosels + dosel_id_y * dose.nb_dosels.x + dosel_id_x;
    if ( density > m_dose_min_density )
    {
        m_dose_values[index] = dose.edep[ index ] / density / vox_vol / gray;
    }
    else
    {
        m_dose_values[index] = 0.0f;
    }

}

/// Class
DoseCalculator::DoseCalculator()
{
    m_dosel_size.x = 0;
    m_dosel_size.y = 0;
    m_dosel_size.z = 0;

    m_offset.x = FLT_MAX;
    m_offset.y = FLT_MAX;
    m_offset.z = FLT_MAX;

    m_nb_of_dosels.x = 0;
    m_nb_of_dosels.y = 0;
    m_nb_of_dosels.z = 0;

    m_xmin = 0; m_xmax = 0;
    m_ymin = 0; m_ymax = 0;
    m_zmin = 0; m_zmax = 0;

    // Min density to compute the dose
    m_dose_min_density = 0.0;

    // Some flags
    m_flag_phantom = false;
    m_flag_materials = false;
    m_flag_dose_calculated = false;
    m_flag_uncertainty_calculated = false;

    // Init the struc
    dose.edep = NULL;
    dose.edep_squared = NULL;
    dose.number_of_hits = NULL;
    dose.nb_dosels.x = 0;
    dose.nb_dosels.y = 0;
    dose.nb_dosels.z = 0;
    dose.dosel_size.x = 0;
    dose.dosel_size.y = 0;
    dose.dosel_size.z = 0;
    dose.inv_dosel_size.x = 0;
    dose.inv_dosel_size.y = 0;
    dose.inv_dosel_size.z = 0;
    dose.offset.x = 0;
    dose.offset.y = 0;
    dose.offset.z = 0;
    dose.xmin = 0; dose.xmax = 0;
    dose.ymin = 0; dose.ymax = 0;
    dose.zmin = 0; dose.zmax = 0;
    dose.tot_nb_dosels = 0;
    dose.slice_nb_dosels = 0;

    // Dose and uncertainty values
    m_dose_values = NULL;
    m_uncertainty_values = NULL;

}

DoseCalculator::~DoseCalculator()
{
    cudaFree( dose.edep );
    cudaFree( dose.edep_squared );
    cudaFree( dose.number_of_hits );

    delete[] m_dose_values;
    delete[] m_uncertainty_values;

/*
    delete [] dose.data_h.edep ;
    delete [] dose.data_h.dose ;
    delete [] dose.data_h.edep_squared ;
    delete [] dose.data_h.number_of_hits ;
    delete [] dose.data_h.uncertainty ;
*/
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

void DoseCalculator::set_dosel_size ( f32 sx, f32 sy, f32 sz )
{
    m_dosel_size = make_f32xyz( sx, sy, sz );
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

VoxVolumeData<f32> * DoseCalculator::get_dose_map()
{
    if ( !m_flag_dose_calculated )
    {
        calculate_dose_to_water();
    }

    VoxVolumeData<f32> *dosemap = new VoxVolumeData<f32>;
    dosemap->nb_vox_x = m_nb_of_dosels.x;
    dosemap->nb_vox_y = m_nb_of_dosels.y;
    dosemap->nb_vox_z = m_nb_of_dosels.z;

    dosemap->off_x = m_offset.x;
    dosemap->off_y = m_offset.y;
    dosemap->off_z = m_offset.z;

    dosemap->spacing_x = m_dosel_size.x;
    dosemap->spacing_y = m_dosel_size.y;
    dosemap->spacing_z = m_dosel_size.z;

    dosemap->number_of_voxels = m_nb_of_dosels.x + m_nb_of_dosels.y + m_nb_of_dosels.z;

    dosemap->xmin = m_xmin;
    dosemap->xmax = m_xmax;
    dosemap->ymin = m_ymin;
    dosemap->ymax = m_ymax;
    dosemap->zmin = m_zmin;
    dosemap->zmax = m_zmax;

    dosemap->values = m_dose_values;

    return dosemap;
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
    if ( m_dosel_size.x > 0.0 && m_dosel_size.y > 0.0 && m_dosel_size.z > 0.0 )
    {
        dose.dosel_size = m_dosel_size;
        dose.inv_dosel_size = fxyz_inv( m_dosel_size );
    }
    else
    {
        dose.dosel_size = make_f32xyz( m_phantom.data_h.spacing_x,
                                       m_phantom.data_h.spacing_y,
                                       m_phantom.data_h.spacing_z );
        dose.inv_dosel_size = fxyz_inv( dose.dosel_size );
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
        dose.xmin = phan_xmin;
        dose.xmax = phan_xmax;
        dose.ymin = phan_ymin;
        dose.ymax = phan_ymax;
        dose.zmin = phan_zmin;
        dose.zmax = phan_zmax;
    }
    else
    {
        dose.xmin = m_xmin;
        dose.xmax = m_xmax;
        dose.ymin = m_ymin;
        dose.ymax = m_ymax;
        dose.zmin = m_zmin;
        dose.zmax = m_zmax;
    }

    // Get the current dimension of the dose map
    f32xyz cur_dose_size = make_f32xyz( dose.xmax - dose.xmin,
                                        dose.ymax - dose.ymin,
                                        dose.zmax - dose.zmin );

    // New nb of voxels
    dose.nb_dosels.x = floor( cur_dose_size.x / dose.dosel_size.x );
    dose.nb_dosels.y = floor( cur_dose_size.y / dose.dosel_size.y );
    dose.nb_dosels.z = floor( cur_dose_size.z / dose.dosel_size.z );
    dose.slice_nb_dosels = dose.nb_dosels.x * dose.nb_dosels.y;
    dose.tot_nb_dosels = dose.slice_nb_dosels * dose.nb_dosels.z;

    // Compute the new size (due to integer nb of doxels)
    f32xyz new_dose_size = fxyz_mul( dose.dosel_size, cast_ui32xyz_to_f32xyz( dose.nb_dosels ) );

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

    dose.xmin += half_delta_size.x;
    dose.xmax -= half_delta_size.x;

    dose.ymin += half_delta_size.y;
    dose.ymax -= half_delta_size.y;

    dose.zmin += half_delta_size.z;
    dose.zmax -= half_delta_size.z;

    // Get the new offset
    dose.offset.x = m_phantom.data_h.off_x - ( dose.xmin - phan_xmin );
    dose.offset.y = m_phantom.data_h.off_y - ( dose.ymin - phan_ymin );
    dose.offset.z = m_phantom.data_h.off_z - ( dose.zmin - phan_zmin );

    //////////////////////////////////////////////////////////

    // Struct allocation
    HANDLE_ERROR( cudaMallocManaged( &(dose.edep), dose.tot_nb_dosels * sizeof( f64 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(dose.edep_squared), dose.tot_nb_dosels * sizeof( f64 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(dose.number_of_hits), dose.tot_nb_dosels * sizeof( ui32 ) ) );

    // Results allocation
    m_dose_values = new f32[ dose.tot_nb_dosels ];
    m_uncertainty_values = new f32[ dose.tot_nb_dosels ];

    // Init values to 0
    for ( ui32 i = 0; i< dose.tot_nb_dosels ; i++ )
    {
        dose.edep[ i ] = 0.0;
        dose.edep_squared[ i ] = 0.0;
        dose.number_of_hits[ i ] = 0.0;

        m_dose_values[ i ] = 0.0;
        m_uncertainty_values[ i ] = 0.0;
    }

/*
    // Copy to GPU if required
    if ( params.data_h.device_target == GPU_DEVICE )
    {
        // GPU allocation
        m_gpu_malloc_dose();
        // Copy data to the GPU
        m_copy_dose_cpu2gpu();
    }
*/
}

void DoseCalculator::calculate_dose_to_water()
{
    GGcout << "Compute dose to water" << GGendl;
    GGcout << GGendl;           

    // Calculate the dose to water and the uncertainty
    for ( ui32 iz=0; iz < dose.nb_dosels.z; iz++ )
    {
        for ( ui32 iy=0; iy < dose.nb_dosels.y; iy++ )
        {
            for ( ui32 ix=0; ix < dose.nb_dosels.x; ix++ )
            {
                m_dose_to_water_calculation( ix, iy, iz );
                m_uncertainty_calculation( ix, iy, iz );
            }
        }
    }

    m_flag_dose_calculated = true;
    m_flag_uncertainty_calculated = true;
}

void DoseCalculator::calculate_dose_to_phantom()
{
    // Check if everything was set properly
    if ( !m_flag_materials || !m_flag_phantom )
    {
        GGcerr << "Dose calculator, phantom and materials data are required!" << GGendl;
        exit_simulation();
    }

    GGcout << "Compute dose to phantom" << GGendl;
    GGcout << GGendl;    

    // Calculate the dose to phantom and the uncertainty
    for ( ui32 iz=0; iz < dose.nb_dosels.z; iz++ )
    {
        for ( ui32 iy=0; iy < dose.nb_dosels.y; iy++ )
        {
            for ( ui32 ix=0; ix < dose.nb_dosels.x; ix++ )
            {
                m_dose_to_phantom_calculation( ix, iy, iz );
                m_uncertainty_calculation( ix, iy, iz );
            }
        }
    }
    
    m_flag_dose_calculated = true;
    m_flag_uncertainty_calculated = true;
}


/*
void DoseCalculator::m_cpu_malloc_dose()
{
    dose.data_h.edep = new f64[dose.data_h.tot_nb_doxels];
    dose.data_h.dose = new f64[dose.data_h.tot_nb_doxels];
    dose.data_h.edep_squared = new f64[dose.data_h.tot_nb_doxels];
    dose.data_h.number_of_hits = new ui32[dose.data_h.tot_nb_doxels];
    dose.data_h.uncertainty = new f64[dose.data_h.tot_nb_doxels];
}
*/
/*
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
*/

/*
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
*/

/*
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
*/


void DoseCalculator::write ( std::string filename )
{
    // Create an IO object
    ImageIO *im_io = new ImageIO;

    std::string format = im_io->get_format( filename );
    filename = im_io->get_filename_without_format( filename );

    // Convert Edep from f64 to f32
    ui32 tot = dose.nb_dosels.x*dose.nb_dosels.y*dose.nb_dosels.z;
    f32 *f32edep = new f32[ tot ];
    ui32 i=0; while ( i < tot )
    {
        f32edep[ i ] = (f32)dose.edep[ i ];
        ++i;
    }

    // Get output name
    std::string edep_out( filename + "-Edep." + format );
    std::string uncer_out( filename + "-Uncertainty." + format );
    std::string hit_out( filename + "-Hit." + format );
    std::string dose_out( filename + "-Dose." + format );

    // Export Edep
    im_io->write_3D( edep_out, f32edep, dose.nb_dosels, dose.offset, dose.dosel_size );

    // Export uncertainty
    if ( !m_flag_uncertainty_calculated )
    {
        // Calculate the dose to phantom and the uncertainty
        for ( ui32 iz=0; iz < dose.nb_dosels.z; iz++ )
        {
            for ( ui32 iy=0; iy < dose.nb_dosels.y; iy++ )
            {
                for ( ui32 ix=0; ix < dose.nb_dosels.x; ix++ )
                {
                    m_uncertainty_calculation( ix, iy, iz );
                }
            }
        }
    }

    // Export uncertainty and hits
    im_io->write_3D( uncer_out, m_uncertainty_values, dose.nb_dosels, dose.offset, dose.dosel_size );
    im_io->write_3D( hit_out, dose.number_of_hits, dose.nb_dosels, dose.offset, dose.dosel_size );

    // Export dose
    if ( m_flag_dose_calculated )
    {
        im_io->write_3D( dose_out, m_dose_values, dose.nb_dosels, dose.offset, dose.dosel_size );
    }

    delete im_io;

}



#endif
