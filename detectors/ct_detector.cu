// GGEMS Copyright (C) 2015

/*!
 * \file ct_detector.cu
 * \brief CT detector (flatpanel)
 * \author Didier Benoit <didier.benoit13@gmail.com>
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.3
 * \date december 2, 2015
 *
 * v0.3: JB - Handle transformation (local frame to global frame) and add unified mem
 * v0.2: JB - Add digitizer
 * v0.1: DB - First code
 */

#ifndef CT_DETECTOR_CU
#define CT_DETECTOR_CU



#include "ct_detector.cuh"

//// GPU functions ///////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ void ct_detector_track_to_in( ParticlesData &particles, ObbData detector_volume,  ui32 id )
{
    // If freeze (not dead), re-activate the current particle
    if( particles.endsimu[ id ] == PARTICLE_FREEZE )
    {
        particles.endsimu[ id ] = PARTICLE_ALIVE;
    }
    else if ( particles.endsimu[ id ] == PARTICLE_DEAD )
    {
        return;
    }

    // Read position
    f32xyz pos;
    pos.x = particles.px[ id ];
    pos.y = particles.py[ id ];
    pos.z = particles.pz[ id ];

    // Read direction
    f32xyz dir;
    dir.x = particles.dx[ id ];
    dir.y = particles.dy[ id ];
    dir.z = particles.dz[ id ];

    // Project particle to detector
//    f32 dist = hit_ray_OBB( pos, dir,
//                            detector_volume.xmin, detector_volume.xmax,
//                            detector_volume.ymin, detector_volume.ymax,
//                            detector_volume.zmin, detector_volume.zmax,
//                            detector_volume.center,
//                            detector_volume.u, detector_volume.v, detector_volume.w );

    //printf(" pos %f %f %f   -   dir %f %f %f\n", pos.x, pos.y, pos.z, dir.x, dir.y, dir.z );

    f32 dist = hit_ray_OBB( pos, dir, detector_volume );

//    printf(" xmin %f xmax %f\n", detector_volume.xmin, detector_volume.xmax );
//    printf(" ymin %f ymax %f\n", detector_volume.ymin, detector_volume.ymax );
//    printf(" zmin %f zmax %f\n", detector_volume.zmin, detector_volume.zmax );
//    printf(" center %f %f %f\n", detector_volume.center.x, detector_volume.center.y, detector_volume.center.z );

    //printf("dist %f\n", dist);

    if( dist == FLT_MAX )
    {
        particles.endsimu[id] = PARTICLE_DEAD;
        particles.E[ id ] = 0.0f;
        return;
    }
    else
    {
        // Check if the path of the particle cross the volume sufficiently
//        f32 cross = dist_overlap_ray_OBB( pos, dir, detector_volume.xmin,
//                                          detector_volume.xmax, detector_volume.ymin, detector_volume.ymax,
//                                          detector_volume.zmin, detector_volume.zmax, detector_volume.center,
//                                          detector_volume.u, detector_volume.v, detector_volume.w );

        f32 cross = dist_overlap_ray_OBB( pos, dir, detector_volume );

//        printf(" Cross %f\n", cross);

        if( cross < EPSILON3 )
        {
            particles.endsimu[id] = PARTICLE_DEAD;
            particles.E[ id ] = 0.0f;
            return;
        }

        // move the particle slightly inside the volume
        pos = fxyz_add( pos, fxyz_scale( dir, dist + EPSILON3 ) );

//        printf(" pos in %f %f %f\n", pos.x, pos.y, pos.z);

    }

    // Save particle position
    particles.px[ id ] = pos.x;
    particles.py[ id ] = pos.y;
    particles.pz[ id ] = pos.z;
}

// Digitizer
__host__ __device__ void ct_detector_digitizer( ParticlesData particles, ObbData detector_volume,
                                                f32xyz pixel_size, ui32xyz nb_pixel,
                                                f32 threshold, f32matrix44 transform,
                                                f32* projection, ui32* scatter_order,
                                                ui8 record_option, bool scatter_option,
                                                ui32 id )
{
    // If freeze or dead, quit
    if( particles.endsimu[ id ] == PARTICLE_FREEZE || particles.endsimu[ id ] == PARTICLE_DEAD )
    {
        return;
    }

    // Read position
    f32xyz pos;
    pos.x = particles.px[ id ];
    pos.y = particles.py[ id ];
    pos.z = particles.pz[ id ];

    pos.x = 321;
    pos.y = 0;
    pos.z = 0;

    // Convert global position into local position
    pos = fxyz_global_to_local_frame( transform, pos );

    printf(" pos %f %f %f\n", pos.x, pos.y, pos.z);

    ui32xyz index = { ui32( pos.x / pixel_size.x ),
                      ui32( pos.y / pixel_size.y ),
                      ui32( pos.z / pixel_size.z ) };

    printf(" ind %i %i %i\n", index.x, index.y, index.z);

    // Check and threshold
    if ( index.x >= nb_pixel.x || index.y >= nb_pixel.y || index.z >= nb_pixel.z || particles.E[ id ] < threshold )
    {
        particles.endsimu[id] = PARTICLE_DEAD;
        particles.E[ id ] = 0.0f;
        return;
    }


    /*
    //f32 rot_posx = pos.x * cosf( orbiting_angle ) + pos.y * sinf( orbiting_angle );  - This var is not used - JB
    f32 rot_posy = -pos.x * sinf( orbiting_angle ) + pos.y * cosf( orbiting_angle );

    // Calculate pixel id
    ui32 idx_xr = (ui32)( ( rot_posy - detector_volume.ymin ) / pixel_size_y );
    ui32 idx_yr = (ui32)( ( pos.z - detector_volume.zmin ) / pixel_size_z );

    if( idx_xr >= nb_pixel_y || idx_yr >= nb_pixel_z || particles.E[ id ] < threshold )
    {
        particles.endsimu[id] = PARTICLE_DEAD;
        particles.E[ id ] = 0.0f;
        return;
    }
    */

    if ( record_option == GET_HIT )
    {
        ggems_atomic_add( projection, index.z * nb_pixel.x*nb_pixel.y +
                          index.y *nb_pixel.x + index.x , 1 );
    }
    else if ( record_option == GET_ENERGY )
    {
        ggems_atomic_add( projection, index.z * nb_pixel.x*nb_pixel.y +
                          index.y *nb_pixel.x + index.x , particles.E[ id ] );
    }


    if ( scatter_option )
    {
        // Scatter increment index
        ui32 n_scatter_order = ( particles.scatter_order[ id ] < MAX_SCATTER_ORDER ) ?
                                 particles.scatter_order[ id ] : MAX_SCATTER_ORDER;

        // Increment the scatter
        if( n_scatter_order != 0 )
        {

            ui32 scatter_idx = ( n_scatter_order - 1 ) * nb_pixel.x * nb_pixel.y * nb_pixel.z;
            ggems_atomic_add( scatter_order, scatter_idx + index.z * nb_pixel.x*nb_pixel.y +
                              index.y * nb_pixel.x + index.x, 1 );
        }
    }


}


// Kernel that move particles to the voxelized volume boundary
__global__ void kernel_ct_detector_track_to_in( ParticlesData particles,
                                                ObbData detector_volume )
{

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    ct_detector_track_to_in( particles, detector_volume, id);
}

// Kernel digitizer
__global__ void kernel_ct_detector_digitizer( ParticlesData particles, ObbData detector_volume,
                                              f32xyz pixel_size, ui32xyz nb_pixel,
                                              f32 threshold, f32matrix44 transform,
                                              f32* projection, ui32* scatter_order,
                                              ui8 record_option, bool scatter_option )
{

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    ct_detector_digitizer( particles, detector_volume,
                           pixel_size, nb_pixel,
                           threshold, transform,
                           projection, scatter_order,
                           record_option, scatter_option,
                           id );
}

//// Class ///////////////////////////////////////////////////////////////////

CTDetector::CTDetector(): GGEMSDetector(),
                          m_threshold( 0.0f ),
                          m_record_option( GET_HIT ),
                          m_record_scatter( false ),
                          m_projection( nullptr ),
                          m_scatter( nullptr )
      //m_projection_h( nullptr ),    // TODO remove
      //m_projection_d( nullptr ),    // TODO remove
      //m_scatter_order_h( nullptr ), // TODO remove
      //m_scatter_order_d( nullptr )  // TODO remove
{
    set_name( "ct_detector" );

    m_pixel_size = make_f32xyz( 0.0, 0.0, 0.0 );
    m_nb_pixel = make_ui32xyz( 0, 0, 0 );
    m_dim = make_f32xyz( 0.0, 0.0, 0.0 );
    m_pos = make_f32xyz( 0.0, 0.0, 0.0 );
    m_angle = make_f32xyz( 0.0, 0.0, 0.0 );

    m_proj_axis.m00 = 0.0;
    m_proj_axis.m01 = 0.0;
    m_proj_axis.m02 = 0.0;
    m_proj_axis.m10 = 0.0;
    m_proj_axis.m11 = 0.0;
    m_proj_axis.m12 = 0.0;
    m_proj_axis.m20 = 0.0;
    m_proj_axis.m21 = 0.0;
    m_proj_axis.m22 = 0.0;
}

CTDetector::~CTDetector()
{
    if( m_projection )
    {
        cudaFree( m_projection );
    }

    if( m_scatter )
    {
        cudaFree( m_scatter );
    }

    /*  TODO: remove
    if( m_projection_h )
    {
        delete[] m_projection_h;
        m_projection_h = nullptr;
    }

    if( m_scatter_order_h )
    {
        delete[] m_scatter_order_h;
        m_scatter_order_h = nullptr;
    }

    if( m_params.data_h.device_target == GPU_DEVICE )
    {
        if( m_projection_d )
        {
            cudaFree( m_projection_d );
        }

        if( m_scatter_order_d )
        {
            cudaFree( m_scatter_order_d );
        }
    }
    */
}

void CTDetector::track_to_in( Particles particles )
{
    if( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id = 0;
        while( id < particles.size )
        {
            ct_detector_track_to_in( particles.data_h,
                                     m_detector_volume,
                                     id );
            ++id;
        }
    }
    else if( m_params.data_h.device_target == GPU_DEVICE )
    {
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 )
                / m_params.data_h.gpu_block_size;

        kernel_ct_detector_track_to_in<<<grid, threads>>>( particles.data_d,
                                                           m_detector_volume );
        cuda_error_check("Error ", " Kernel_ct_detector (track to in)");
        cudaThreadSynchronize();
    }
}

void CTDetector::digitizer( Particles particles )
{
    if( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id = 0;
        while( id < particles.size )
        {
            ct_detector_digitizer( particles.data_h, m_detector_volume,
                                   m_pixel_size, m_nb_pixel,
                                   m_threshold, m_transform,
                                   m_projection, m_scatter,
                                   m_record_option, m_record_scatter, id );
            ++id;
        }
    }
    else if( m_params.data_h.device_target == GPU_DEVICE )
    {
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 )
                / m_params.data_h.gpu_block_size;

        kernel_ct_detector_digitizer<<<grid, threads>>>( particles.data_d, m_detector_volume,
                                                         m_pixel_size, m_nb_pixel,
                                                         m_threshold, m_transform,
                                                         m_projection, m_scatter,
                                                         m_record_option, m_record_scatter );
        cuda_error_check("Error ", " Kernel_ct_detector (digitizer)");
        cudaThreadSynchronize();
    }
}

void CTDetector::save_projection( std::string filename, std::string format )
{
    // Check format in bit: 16 or 32
    if ( format != "ui16" && format != "ui32" && format != "f32" )
    {
        GGcerr << "Image projection must have one format from this list: 'ui16', 'ui32' or 'f32': " << format << " given!" << GGendl;
        GGwarn << "Projection will be then exported in f32." << GGendl;
        format = "f32";
    }

    // First compute some parameters of the projection
    f32xyz offset = make_f32xyz( 0.5 * m_pixel_size.x * m_nb_pixel.x,
                                 0.5 * m_pixel_size.y * m_nb_pixel.y,
                                 0.5 * m_pixel_size.z * m_nb_pixel.z );

    // Create IO object
    ImageIO *im_io = new ImageIO;

    /*
    // Check if CPU or GPU
    if( m_params.data_h.device_target == GPU_DEVICE )
    {
        HANDLE_ERROR( cudaMemcpy( m_projection_h,
                                  m_projection_d,
                                  sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z,
                                  cudaMemcpyDeviceToHost ) );
    }
    */

    // If ui16 format, need to convert the data
    if ( format == "ui16" )
    {
        ui16 *projection16 = new ui16[ m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z ];
        for( ui32 i = 0; i < m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z; ++i )
        {
            projection16[ i ] = (ui16)m_projection[ i ];
        }
        im_io->write_3D( filename, projection16, m_nb_pixel, offset, m_pixel_size );
        delete[] projection16;
    }

    // The same for ui32 format
    if ( format == "ui32" )
    {
        ui32 *projection32 = new ui32[ m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z ];
        for( ui32 i = 0; i < m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z; ++i )
        {
            projection32[ i ] = (ui32)m_projection[ i ];
        }
        im_io->write_3D( filename, projection32, m_nb_pixel, offset, m_pixel_size );
        delete[] projection32;
    }

    // No need conversion for f32 format
    if ( format == "f32" )
    {
        im_io->write_3D( filename, m_projection, m_nb_pixel, offset, m_pixel_size );
    }


//    f32xy offset, spacing;
//    ui32xy nb_pix;

//    // Proj YZ
//    if ( m_nb_pixel_x == 1 )
//    {
//        offset = make_f32xy( 0.5*m_pixel_size_y*m_nb_pixel_y, 0.5*m_pixel_size_z*m_nb_pixel_z );
//        spacing = make_f32xy( m_pixel_size_y, m_pixel_size_z );
//        nb_pix = make_ui32xy( m_nb_pixel_y, m_nb_pixel_z );
//    }
//    // Proj XZ
//    else if ( m_nb_pixel_y == 1 )
//    {
//        offset = make_f32xy( 0.5*m_pixel_size_x*m_nb_pixel_x, 0.5*m_pixel_size_z*m_nb_pixel_z );
//        spacing = make_f32xy( m_pixel_size_x, m_pixel_size_z );
//        nb_pix = make_ui32xy( m_nb_pixel_x, m_nb_pixel_z );
//    }
//    // Proj XY
//    else if ( m_nb_pixel_z == 1 )
//    {
//        offset = make_f32xy( 0.5*m_pixel_size_x*m_nb_pixel_x, 0.5*m_pixel_size_y*m_nb_pixel_y );
//        spacing = make_f32xy( m_pixel_size_x, m_pixel_size_y );
//        nb_pix = make_ui32xy( m_nb_pixel_x, m_nb_pixel_y );
//    }
//    // ERROR: If not a 2D projection, export in 3D
//    else
//    {
//        GGcerr << "For exporting 2D projection Image must have one of its dimension equal to 1, current dimension: "
//               << m_nb_pixel_x << "x" <<  m_nb_pixel_y << "x" << m_nb_pixel_z
//               << GGendl;
//        return;
//        /*
//        f32xyz offset3D = make_f32xyz( 0.5*m_pixel_size_x*m_nb_pixel_x, 0.5*m_pixel_size_y*m_nb_pixel_y, 0.5*m_pixel_size_z*m_nb_pixel_z );
//        f32xyz spacing3D = make_f32xyz( m_pixel_size_x, m_pixel_size_y, m_pixel_size_z );
//        ui32xyz nb_pix3D = make_ui32xyz( m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z );

//        if ( format == "16" )
//        {
//            im_io->write_3D( filename, projection16, nb_pix3D, offset3D, spacing3D );
//            delete[] projection16;
//        }
//        else if ( format == "32" )
//        {
//            im_io->write_3D( filename, m_projection_h, nb_pix3D, offset3D, spacing3D );
//        }

//        delete im_io;
//        return;
//        */
//    }

//    // Export the projection
//    if ( format == "ui16" )
//    {
//        im_io->write_2D( filename, projection16, nb_pix, offset, spacing );
//        delete[] projection16;
//    }
//    else if ( format == "ui32" )
//    {
//        im_io->write_2D( filename, m_projection_h, nb_pix, offset, spacing );
//    }

    delete im_io;
}

//            // Global sinogram
//            ImageReader::record3Dimage(
//                        filename,
//                        projection16,
//                        make_f32xyz( 0.0f, 0.0f, 0.0f ),
//                        make_f32xyz( m_pixel_size_x, m_pixel_size_y, m_pixel_size_z ),
//                        make_ui32xyz( m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z ),
//                        false
//                        );

void CTDetector::save_scatter( std::string filename )
{
    // Check
    if ( !m_record_scatter )
    {
        GGwarn << "CTDetector, nothing to export, the recording scatter was not requested" << GGendl;
        return;
    }

    // Create io object
    ImageIO *im_io = new ImageIO;

    // Check format
    std::string basename = im_io->get_filename_without_extension( filename );

    /*
    // GPU
    if( m_params.data_h.device_target == GPU_DEVICE )
    {
        HANDLE_ERROR( cudaMemcpy( m_scatter_order_h,
                                  m_scatter_order_d,
                                  sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z
                                  * MAX_SCATTER_ORDER, cudaMemcpyDeviceToHost ) );
    }
    */

    // First compute some parameters of the projection
    f32xyz offset = make_f32xyz( 0.5 * m_pixel_size.x * m_nb_pixel.x,
                                 0.5 * m_pixel_size.y * m_nb_pixel.y,
                                 0.5 * m_pixel_size.z * m_nb_pixel.z );

    // Allocation
    ui32 tot_pix = m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z;
    ui16 *scatter16 = new ui16[ tot_pix ];

//    // Determine the orientation of the projection
//    f32xy offset, spacing;
//    ui32xy nb_pix;

//    // Proj YZ
//    if ( m_nb_pixel_x == 1 )
//    {
//        offset = make_f32xy( 0.5*m_pixel_size_y*m_nb_pixel_y, 0.5*m_pixel_size_z*m_nb_pixel_z );
//        spacing = make_f32xy( m_pixel_size_y, m_pixel_size_z );
//        nb_pix = make_ui32xy( m_nb_pixel_y, m_nb_pixel_z );
//    }
//    // Proj XZ
//    else if ( m_nb_pixel_y == 1 )
//    {
//        offset = make_f32xy( 0.5*m_pixel_size_x*m_nb_pixel_x, 0.5*m_pixel_size_z*m_nb_pixel_z );
//        spacing = make_f32xy( m_pixel_size_x, m_pixel_size_z );
//        nb_pix = make_ui32xy( m_nb_pixel_x, m_nb_pixel_z );
//    }
//    // Proj XY
//    else if ( m_nb_pixel_z == 1 )
//    {
//        offset = make_f32xy( 0.5*m_pixel_size_x*m_nb_pixel_x, 0.5*m_pixel_size_y*m_nb_pixel_y );
//        spacing = make_f32xy( m_pixel_size_x, m_pixel_size_y );
//        nb_pix = make_ui32xy( m_nb_pixel_x, m_nb_pixel_y );
//    }
//    // ERROR: If not a 2D projection, export in 3D
//    else
//    {
//        GGcerr << "For exporting 2D scatter projection Image must have one of its dimension equal to 1, current dimension: "
//               << m_nb_pixel_x << "x" <<  m_nb_pixel_y << "x" << m_nb_pixel_z
//               << GGendl;
//        return;
//    }

    // Loop over the scatter order
    for( ui32 i = 0; i < MAX_SCATTER_ORDER; ++i )
    {
        // Determine the filename
        std::ostringstream out( std::ostringstream::out );
        out << basename << "_" << std::setfill( '0' ) << std::setw( 3 ) << i
            << ".mhd";

        // Get the corresponding scatter image
        for( ui32 j = 0; j < tot_pix; ++j )
        {
            scatter16[ j ] = (ui16)m_scatter[ i*tot_pix + j ];
        }

        // Record the projection
        im_io->write_3D( out.str(), scatter16, m_nb_pixel, offset, m_pixel_size );

//        // Save the scatter image for each order
//        ImageReader::record3Dimage(
//                    out.str(),
//                    &scatter16[ i * m_nb_pixel_x * m_nb_pixel_y ],
//                make_f32xyz( 0.0f, 0.0f, 0.0f ),
//                make_f32xyz( m_pixel_size_x, m_pixel_size_y, m_pixel_size_z ),
//                make_ui32xyz( m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z ),
//                false
//                );

    }
    delete[] scatter16;

    delete im_io;
}




//// Setting ////////////////////////////////////////////////////////////

//void CTDetector::set_dimension( f32 sizex, f32 sizey, f32 sizez )
//{
//    m_dim = make_f32xyz( sizex, sizey, sizez );
//}

void CTDetector::set_pixel_size( f32 sx, f32 sy, f32 sz )
{
    m_pixel_size = make_f32xyz( sx, sy, sz );
}

void CTDetector::set_number_of_pixels(ui32 nx, ui32 ny , ui32 nz )
{
    m_nb_pixel = make_ui32xyz( nx, ny, nz );
}

void CTDetector::set_position( f32 x, f32 y, f32 z )
{
    m_pos = make_f32xyz( x, y, z );
}

void CTDetector::set_threshold( f32 threshold )
{
    m_threshold = threshold;
}

void CTDetector::set_rotation( f32 rx, f32 ry, f32 rz )
{
    m_angle = make_f32xyz( rx, ry, rz );
}

// Setting the axis transformation matrix
void CTDetector::set_projection_axis( f32 m00, f32 m01, f32 m02,
                                      f32 m10, f32 m11, f32 m12,
                                      f32 m20, f32 m21, f32 m22 )
{
    m_proj_axis.m00 = m00;
    m_proj_axis.m01 = m01;
    m_proj_axis.m02 = m02;
    m_proj_axis.m10 = m10;
    m_proj_axis.m11 = m11;
    m_proj_axis.m12 = m12;
    m_proj_axis.m20 = m20;
    m_proj_axis.m21 = m21;
    m_proj_axis.m22 = m22;
}

// Setting record option
void CTDetector::set_record_option( std::string opt )
{
    // Transform the name of the option in small letter
    std::transform( opt.begin(), opt.end(), opt.begin(), ::tolower );

    if( opt == "hits" )
    {
        m_record_option = GET_HIT;
    }
    else if ( opt == "energies" )
    {
        m_record_option = GET_ENERGY;
    }
    else
    {
        GGcerr << "CTDetector, record option unknow: " << opt << GGendl;
        exit_simulation();
    }
}

// Setting scatter option
void CTDetector::set_record_scatter( bool flag )
{
    m_record_scatter = flag;
}

/*
bool CTDetector::m_check_mandatory()
{
    if( m_pixel_size_x == 0.0f ||
            m_pixel_size_y == 0.0f ||
            m_pixel_size_z == 0.0f ||
            m_nb_pixel_x == 0 ||
            m_nb_pixel_y == 0 ||
            m_nb_pixel_z == 0 )
    {
        return false;
    }
    else
    {
        return true;
    }
}
*/

//// Getting functions ///////////////////////////////////////////////

f32matrix44 CTDetector::get_transformation()
{
    return m_transform;
}

ObbData CTDetector::get_bounding_box()
{
    return m_detector_volume;
}

//// Private functions ///////////////////////////////////////////////

ui32 CTDetector::get_detected_particles()
{
    ui32 count = 0;
    for( ui32 i = 0; i < m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z; ++i )
    {
        count += m_projection[ i ];
    }

    return count;
}

ui32 CTDetector::get_scatter_number( ui32 scatter_order )
{

    if ( m_record_scatter )
    {
        ui32 count = 0;
        for( ui32 i = 0; i < m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z; ++i )
        {
            count += m_scatter[ i + m_nb_pixel.x * m_nb_pixel.y * m_nb_pixel.z * scatter_order ];
        }

        return count;
    }
    else
    {
        GGwarn << "Recording of the scatter was not request" << GGendl;
        return 0;
    }

}

/* TOD: to review
void CTDetector::print_info_scatter()
{
    // Get the number of detected particles
    ui32 detected_particles = getDetectedParticles();

    // Get the number of scatter by order and the total scatter
    ui32 count_scatter[ MAX_SCATTER_ORDER ];
    ui32 total_scatter = 0;
    for( ui32 i = 0; i < MAX_SCATTER_ORDER; ++i )
    {
        count_scatter[ i ] = getScatterNumber( i );
        total_scatter += count_scatter[ i ];
    }

    // Direct particle
    ui32 direct_particles = detected_particles - total_scatter;

    std::cout << std::endl;
    GGcout << "------------------------------------------" << GGendl;
    GGcout << "Detected particles:            " << std::setfill( ' ' )
           << std::setw( 10 ) << detected_particles << GGendl;
    GGcout << "Direct particles:              " << std::setfill( ' ' )
           << std::setw( 10 ) << direct_particles << " [" << std::setfill( ' ' )
           << std::setw( 6 ) << std::setprecision( 2 ) << std::fixed
           << 100.0f * (float)direct_particles / detected_particles << " %]" << GGendl;
    for( ui32 i = 0; i < MAX_SCATTER_ORDER; ++i )
    {
        GGcout << "Scattered particles order " << std::setfill( ' ' )
               << std::setw( 2 ) << i + 1 << " : " << std::setfill( ' ' )
               << std::setw( 10 ) << count_scatter[ i ] << " [" << std::setfill( ' ' )
               << std::setw( 6 ) << std::setprecision( 2 ) << std::fixed
               << 100.0f * (float)count_scatter[ i ] / detected_particles << " %]"
               << GGendl;
    }
    std::cout << std::endl;

}
*/

/* TODO: remove
void CTDetector::m_copy_detector_cpu2gpu()
{
    m_detector_volume.volume.data_d.xmin = m_detector_volume.volume.data_h.xmin;
    m_detector_volume.volume.data_d.xmax = m_detector_volume.volume.data_h.xmax;

    m_detector_volume.volume.data_d.ymin = m_detector_volume.volume.data_h.ymin;
    m_detector_volume.volume.data_d.ymax = m_detector_volume.volume.data_h.ymax;

    m_detector_volume.volume.data_d.zmin = m_detector_volume.volume.data_h.zmin;
    m_detector_volume.volume.data_d.zmax = m_detector_volume.volume.data_h.zmax;

    m_detector_volume.volume.data_d.angle = m_detector_volume.volume.data_h.angle;
    m_detector_volume.volume.data_d.translate = m_detector_volume.volume.data_h.translate;
    m_detector_volume.volume.data_d.center = m_detector_volume.volume.data_h.center;

    m_detector_volume.volume.data_d.u = m_detector_volume.volume.data_h.u;
    m_detector_volume.volume.data_d.v = m_detector_volume.volume.data_h.v;
    m_detector_volume.volume.data_d.w = m_detector_volume.volume.data_h.w;

    m_detector_volume.volume.data_d.size = m_detector_volume.volume.data_h.size;
}
*/

void CTDetector::initialize( GlobalSimulationParameters params )
{
    // Check the parameters
    if ( m_pixel_size.x == 0.0 || m_pixel_size.y == 0.0 || m_pixel_size.z == 0.0 ||
         m_nb_pixel.x == 0 || m_nb_pixel.y == 0 || m_nb_pixel.z == 0 )
    {
        GGcerr << "CTDetector: one of the pixel sizes or nb of pixel parameters is missing!" << GGendl;
        exit_simulation();
    }

    // Params
    m_params = params;

    // Compute the transformation matrix of the detector that map local frame to glboal frame
    TransformCalculator *trans = new TransformCalculator;
    trans->set_translation( m_pos.x, m_pos.y, m_pos.z );
    trans->set_rotation( m_angle );
    trans->set_axis_transformation( m_proj_axis );
    m_transform = trans->get_transformation_matrix();
    delete trans;

    /// Defining an OBB for the flat panel (local frame)

    // Get dimension of the flatpanel in local coordinate
    m_dim.x = m_pixel_size.x * m_nb_pixel.x;
    m_dim.y = m_pixel_size.y * m_nb_pixel.y;
    m_dim.z = m_pixel_size.z * m_nb_pixel.z;

    m_detector_volume.xmin = 0;
    m_detector_volume.xmax = m_dim.x;
    m_detector_volume.ymin = 0;
    m_detector_volume.ymax = m_dim.y;
    m_detector_volume.zmin = 0;
    m_detector_volume.zmax = m_dim.z;

    // Store the matrix
    m_detector_volume.transformation = m_transform;

//    // Compute the OBB axis (global frame)
//    f32xyz uloc = { 1.0, 0.0, 0.0 }; // x
//    f32xyz vloc = { 0.0, 1.0, 0.0 }; // y
//    f32xyz wloc = { 0.0, 0.0, 1.0 }; // z
//    m_detector_volume.u = fxyz_local_to_global_frame( m_transform, uloc );
//    m_detector_volume.v = fxyz_local_to_global_frame( m_transform, vloc );
//    m_detector_volume.w = fxyz_local_to_global_frame( m_transform, wloc );

    // Allocation
    HANDLE_ERROR( cudaMallocManaged( &m_projection,  m_nb_pixel.x*m_nb_pixel.y*m_nb_pixel.z * sizeof( f32 ) ) );
    if ( m_record_scatter )
    {
        HANDLE_ERROR( cudaMallocManaged( &m_scatter,  MAX_SCATTER_ORDER *
                                         m_nb_pixel.x*m_nb_pixel.y*m_nb_pixel.z * sizeof( ui32 ) ) );
    }

// TODO: remove
//    // Copy projection data to GPU
//    if( m_params.data_h.device_target == GPU_DEVICE )
//    {
//        m_copy_detector_cpu2gpu();

//        // GPU mem allocation
//        HANDLE_ERROR( cudaMalloc( (void**)&m_projection_d,
//                                  m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z * sizeof( ui32 ) ) );
//        // GPU mem copy
//        HANDLE_ERROR( cudaMemcpy( m_projection_d,
//                                  m_projection_h,
//                                  sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z,
//                                  cudaMemcpyHostToDevice ) );

//        // GPU mem allocation
//        HANDLE_ERROR( cudaMalloc( (void**)&m_scatter_order_d,
//                                  MAX_SCATTER_ORDER * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z
//                                  * sizeof( ui32 ) ) );
//        // GPU mem copy
//        HANDLE_ERROR( cudaMemcpy( m_scatter_order_d,
//                                  m_scatter_order_h,
//                                  MAX_SCATTER_ORDER * sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y
//                                  * m_nb_pixel_z, cudaMemcpyHostToDevice ) );
//    }
}

#endif

