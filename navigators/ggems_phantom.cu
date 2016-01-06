// GGEMS Copyright (C) 2015

/*!
 * \file ggems_phantom.cu
 * \brief Abstract source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Abstract class that handle every sources used in GGEMS
 *
 */

#ifndef GGEMS_PHANTOM_CU
#define GGEMS_PHANTOM_CU

#include "ggems_phantom.cuh"

GGEMSPhantom::GGEMSPhantom()
{
    m_phantom_name = "no_source";
}

void GGEMSPhantom::set_name ( std::string name )
{
    m_phantom_name = name;
}

std::string GGEMSPhantom::get_name()
{
    return m_phantom_name;
}


#endif
