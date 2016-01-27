// GGEMS Copyright (C) 2015

/*!
 * \file ct_detector.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef CT_DETECTOR_CU
#define CT_DETECTOR_CU

#define MAX_SCATTER_ORDER 3

#include <iomanip>
#include <sstream>

#include "ggems_detector.cuh"
#include "ct_detector.cuh"
#include "image_reader.cuh"

__host__ __device__ void ct_detector_track_to_in( ParticlesData &particles,
  ObbData detector_volume, ui32* projection, ui32* scatter_order,
  f32 pixel_size_x, f32 pixel_size_y, f32 pixel_size_z, ui32 nb_pixel_x,
  ui32 nb_pixel_y, ui32 nb_pixel_z, f32 pos_x, f32 pos_y, f32 pos_z,
  f32 threshold, ui32 id )
{
  // If freeze (not dead), re-activate the current particle
  if( particles.endsimu[ id ] == PARTICLE_FREEZE )
  {
    particles.endsimu[ id ] = PARTICLE_ALIVE;
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
  f32 dist = hit_ray_OBB( pos, dir,
    detector_volume.xmin, detector_volume.xmax,
    detector_volume.ymin, detector_volume.ymax,
    detector_volume.zmin, detector_volume.zmax,
    detector_volume.center,
    detector_volume.u, detector_volume.v, detector_volume.w );

  /*printf( "x: %4.7f %4.7f\n", detector_volume.xmin, detector_volume.xmax );
  printf( "y: %4.7f %4.7f\n", detector_volume.ymin, detector_volume.ymax );
  printf( "z: %4.7f %4.7f\n", detector_volume.zmin, detector_volume.zmax );
  printf( "centerx: %4.7f %4.7f %4.7f\n", detector_volume.center.x, detector_volume.center.y, detector_volume.center.z );
  printf( "u: %4.7f %4.7f %4.7f\n", detector_volume.u.x, detector_volume.u.y, detector_volume.u.z );
  printf( "v: %4.7f %4.7f %4.7f\n", detector_volume.v.x, detector_volume.v.y, detector_volume.v.z );
  printf( "w: %4.7f %4.7f %4.7f\n", detector_volume.w.x, detector_volume.w.y, detector_volume.w.z );
  printf( "angle: %4.7f %4.7f %4.7f\n", detector_volume.angle.x, detector_volume.angle.y, detector_volume.angle.z );*/

  if( dist == FLT_MAX )
  {
    particles.endsimu[id] = PARTICLE_DEAD;
    particles.E[ id ] = 0.0f;
    return;
  }
  else
  {
    // Check if the path of the particle cross the volume sufficiently
    f32 cross = dist_overlap_ray_OBB( pos, dir, detector_volume.xmin,
      detector_volume.xmax, detector_volume.ymin, detector_volume.ymax,
      detector_volume.zmin, detector_volume.zmax, detector_volume.center,
      detector_volume.u, detector_volume.v, detector_volume.w );

    if( cross < EPSILON3 ) 
    {
      particles.endsimu[id] = PARTICLE_DEAD;
      particles.E[ id ] = 0.0f;
      return;
    }

    // move the particle slightly inside the volume
    pos = fxyz_add( pos, fxyz_scale( dir, dist + EPSILON3 ) );
  }

  // Calculate pixel id
  ui32 idx_xr = (ui32)( ( pos.y - detector_volume.ymin ) / pixel_size_y );
  ui32 idx_yr = nb_pixel_z - 1 - (ui32)( ( pos.z - detector_volume.zmin ) / pixel_size_z );

  if( idx_xr >= nb_pixel_y || idx_yr >= nb_pixel_z )
  {
    particles.endsimu[id] = PARTICLE_DEAD;
    particles.E[ id ] = 0.0f;
    return;
  }

  if( particles.E[ id ] < threshold )
  {
    particles.endsimu[id] = PARTICLE_DEAD;
    particles.E[ id ] = 0.0f;
    return;
  }

  ggems_atomic_add( projection, idx_xr + idx_yr * nb_pixel_y, 1 );

  // Scatter increment index
  ui32 n_scatter_order = ( particles.scatter_order[ id ] < MAX_SCATTER_ORDER ) ?
    particles.scatter_order[ id ] : MAX_SCATTER_ORDER;

  // Increment the scatter
  if( n_scatter_order != 0 )
  {
    ui32 scatter_idx =
      ( n_scatter_order - 1 ) * nb_pixel_y * nb_pixel_z;
    ggems_atomic_add( scatter_order,
      idx_xr + idx_yr * nb_pixel_y + scatter_idx, 1 );
  }
}

// Kernel that move particles to the voxelized volume boundary
__global__ void kernel_ct_detector_track_to_in( ParticlesData particles,
  ObbData detector_volume, ui32* projection, ui32* scatter_order,
  f32 pixel_size_x, f32 pixel_size_y, f32 pixel_size_z, ui32 nb_pixel_x,
  ui32 nb_pixel_y, ui32 nb_pixel_z, f32 pos_x, f32 pos_y, f32 pos_z,
  f32 threshold )
{

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    ct_detector_track_to_in( particles, detector_volume, projection,
      scatter_order, pixel_size_x, pixel_size_y, pixel_size_z, nb_pixel_x,
      nb_pixel_y, nb_pixel_z, pos_x, pos_y, pos_z, threshold, id);
}

void CTDetector::track_to_in( Particles particles )
{
  if( m_params.data_h.device_target == CPU_DEVICE )
  {
    ui32 id = 0;
    while( id < particles.size )
    {
      ct_detector_track_to_in( particles.data_h,
        m_detector_volume.volume.data_h,
        m_projection_h, m_scatter_order_h,
        m_pixel_size_x, m_pixel_size_y, m_pixel_size_z,
        m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z,
        m_posx, m_posy, m_posz,
        m_threshold,
        id
      );
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
      m_detector_volume.volume.data_d,
      m_projection_d, m_scatter_order_d,
      m_pixel_size_x, m_pixel_size_y, m_pixel_size_z,
      m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z,
      m_posx, m_posy, m_posz,
      m_threshold
    );
    cuda_error_check("Error ", " Kernel_ct_detector (track to in)");
    cudaThreadSynchronize();
  }
}

CTDetector::CTDetector()
: GGEMSDetector(),
  m_pixel_size_x( 0.0f ),
  m_pixel_size_y( 0.0f ),
  m_pixel_size_z( 0.0f ),
  m_nb_pixel_x( 0 ),
  m_nb_pixel_y( 0 ),
  m_nb_pixel_z( 0 ),
  m_posx( 0.0f ),
  m_posy( 0.0f ),
  m_posz( 0.0f ),
  m_threshold( 0.0f ),
  m_projection_h( nullptr ),
  m_projection_d( nullptr ),
  m_scatter_order_h( nullptr ),
  m_scatter_order_d( nullptr )
{
  set_name( "ct_detector" );
}

CTDetector::~CTDetector()
{
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
}

void CTDetector::set_dimension( f32 w, f32 h, f32 d )
{
  m_nb_pixel_x = w;
  m_nb_pixel_y = h;
  m_nb_pixel_z = d;
}

void CTDetector::set_pixel_size( f32 sx, f32 sy, f32 sz )
{
  m_pixel_size_x = sx;
  m_pixel_size_y = sy;
  m_pixel_size_z = sz;
}

void CTDetector::set_position( f32 x, f32 y, f32 z )
{
  m_posx = x;
  m_posy = y;
  m_posz = z;
}

void CTDetector::set_threshold( f32 threshold )
{
  m_threshold = threshold;
}

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

void CTDetector::save_projection( std::string filename )
{
  // Check if CPU or GPU
  if( m_params.data_h.device_target == GPU_DEVICE )
  {
    HANDLE_ERROR( cudaMemcpy( m_projection_h,
      m_projection_d,
      sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z,
      cudaMemcpyDeviceToHost ) );
  }

  // Global sinogram
  ImageReader::record3Dimage(
    filename,
    m_projection_h,
    make_f32xyz( 0.0f, 0.0f, 0.0f ),
    make_f32xyz( m_pixel_size_x, m_pixel_size_y, m_pixel_size_z ),
    make_i32xyz( m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z ),
    false
  );

}

void CTDetector::save_scatter( std::string basename )
{
  if( m_params.data_h.device_target == GPU_DEVICE )
  {
    HANDLE_ERROR( cudaMemcpy( m_scatter_order_h,
      m_scatter_order_d,
      sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z
      * MAX_SCATTER_ORDER, cudaMemcpyDeviceToHost ) );
  }

  // Loop over the scatter order
  for( ui32 i = 0; i < MAX_SCATTER_ORDER; ++i )
  {
    // Determine the filename
    std::ostringstream out( std::ostringstream::out );
    out << basename << "_" << std::setfill( '0' ) << std::setw( 3 ) << i
      << ".mhd";

    // Save the scatter image for each order
    ImageReader::record3Dimage(
      out.str(),
      &m_scatter_order_h[ i * m_nb_pixel_x * m_nb_pixel_y ],
      make_f32xyz( 0.0f, 0.0f, 0.0f ),
      make_f32xyz( m_pixel_size_x, m_pixel_size_y, m_pixel_size_z ),
      make_i32xyz( m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z ),
      false
    );
  }
}

ui32 CTDetector::getDetectedParticles()
{
  ui32 count = 0;
  for( ui32 i = 0; i < m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z; ++i )
  {
    count += m_projection_h[ i ];
  }

  return count;
}

ui32 CTDetector::getScatterNumber( ui32 scatter_order )
{
  ui32 count = 0;
  for( ui32 i = 0; i < m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z; ++i )
  {
    count += m_scatter_order_h[ i + m_nb_pixel_x * m_nb_pixel_y
      * m_nb_pixel_z * scatter_order ];
  }

  return count;
}

void CTDetector::printInfoDetection()
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

void CTDetector::initialize( GlobalSimulationParameters params )
{
  // Check the parameters
  if( !m_check_mandatory() )
  {
    print_error( "CTDetector: missing parameters!!!" );
    exit_simulation();
  }

  // Params
  m_params = params;

  // Fill the detector volume parameters
  m_detector_volume.set_size(
    m_pixel_size_x * m_nb_pixel_x,
    m_pixel_size_y * m_nb_pixel_y,
    m_pixel_size_z * m_nb_pixel_z
  );

  m_detector_volume.set_center_position( 0.0f, 0.0f, 0.0f );
  m_detector_volume.translate( m_posx, m_posy, m_posz );

  // Allocate
  m_projection_h = new ui32[ m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z ];
  memset( m_projection_h, 0.0f, m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z
    * sizeof( ui32 ) );

  // MAX_SCATTER_ORDER first scatter orders are only registered
  m_scatter_order_h =
    new ui32[ MAX_SCATTER_ORDER * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z ];
  memset( m_scatter_order_h, 0.0f, MAX_SCATTER_ORDER * m_nb_pixel_x
    * m_nb_pixel_y * m_nb_pixel_z * sizeof( ui32 ) );

  // Copy projection data to GPU
  if( m_params.data_h.device_target == GPU_DEVICE )
  {
    m_copy_detector_cpu2gpu();

    // GPU mem allocation
    HANDLE_ERROR( cudaMalloc( (void**)&m_projection_d,
      m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z * sizeof( ui32 ) ) );
    // GPU mem copy
    HANDLE_ERROR( cudaMemcpy( m_projection_d,
      m_projection_h,
      sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z,
      cudaMemcpyHostToDevice ) );

    // GPU mem allocation
    HANDLE_ERROR( cudaMalloc( (void**)&m_scatter_order_d,
      MAX_SCATTER_ORDER * m_nb_pixel_x * m_nb_pixel_y * m_nb_pixel_z
      * sizeof( ui32 ) ) );
    // GPU mem copy
    HANDLE_ERROR( cudaMemcpy( m_scatter_order_d,
      m_scatter_order_h,
      MAX_SCATTER_ORDER * sizeof( ui32 ) * m_nb_pixel_x * m_nb_pixel_y
      * m_nb_pixel_z, cudaMemcpyHostToDevice ) );
  }
}

#endif

