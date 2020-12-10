#ifndef GUARD_GGEMS_IO_GGEMSHISTOGRAMMODE_HH
#define GUARD_GGEMS_IO_GGEMSHISTOGRAMMODE_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSHistogramMode.hh

  \brief Structure storing histogram infos

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday December 3, 2020
*/

#include <memory>

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSHistogramMode_t
  \brief Structure storing histogram infos
*/
typedef struct GGEMSHistogramMode_t
{
  std::shared_ptr<cl::Buffer> histogram_cl_; /*!< Buffer storing histogram on OpenCL device */
  GGint number_of_elements_; /*!< Number of elements in hit buffer */
} GGEMSHistogramMode; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // End of GUARD_GGEMS_IO_GGEMSHISTOGRAMMODE_HH
