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

#include "ggems_detector.cuh"
#include "ct_detector.cuh"

CTDetector::CTDetector()
: GGEMSDetector(),
  m_pixel_size_x( 0.0f ),
  m_pixel_size_y( 0.0f ),
  m_pixel_size_z( 0.0f ),
  m_nb_pixel_x( 0 ),
  m_nb_pixel_y( 0 ),
  m_orbiting_radius( 0 ),
  m_projection_h( nullptr ),
  m_projection_d( nullptr )
{
  ;
}

void CTDetector::set_width( f32 w )
{
  m_nb_pixel_x = w;
}

void CTDetector::set_height( f32 h )
{
  m_nb_pixel_y = h;
}

void CTDetector::set_pixel_size( f32 sx, f32 sy, f32 sz )
{
  m_pixel_size_x = sx;
  m_pixel_size_y = sy;
  m_pixel_size_z = sz;
}

void CTDetector::set_orbiting_radius( f32 r )
{
  m_orbiting_radius = r;
}

void CTDetector::initialize( GlobalSimulationParameters params )
{
  ;
}

#endif

