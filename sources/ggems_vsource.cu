// GGEMS Copyright (C) 2015

/*!
 * \file ggems_vsource.cu
 * \brief Virtual source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Virtual class that handle every sources used in GGEMS
 *
 */

#ifndef GGEMS_VSOURCE_CU
#define GGEMS_VSOURCE_CU

#include "ggems_vsource.cuh"

GGEMSVSource::GGEMSVSource() {}
GGEMSVSource::~GGEMSVSource() {}

virtual void GGEMSVSource::get_primaries_generator(ParticleStack particles) {}
virtual void GGEMSVSource::initialize(GlobalSimulationParameters params) {}

#endif
