// GGEMS Copyright (C) 2015

/*!
 * \file ggems_source.cu
 * \brief Abstract source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Abstract class that handle every sources used in GGEMS
 *
 */

#ifndef GGEMS_SOURCE_CU
#define GGEMS_SOURCE_CU

#include "ggems_source.cuh"

GGEMSSource::GGEMSSource()
: m_source_name( "no_source" )
{
    ;
}

void GGEMSSource::set_name(std::string name) {
    m_source_name = name;
}

#endif
