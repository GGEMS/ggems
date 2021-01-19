#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSDOSERECORDING_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSDOSERECORDING_HH

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
  \file GGEMSDoseRecording.hh

  \brief Structure storing histogram infos

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday January 19, 2021
*/

#include <memory>

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSDoseRecording_t
  \brief Structure storing data for dose recording
*/
typedef struct GGEMSDoseRecording_t
{
  std::shared_ptr<cl::Buffer> edep_; /*!< Buffer storing energy deposit on OpenCL device */
  std::shared_ptr<cl::Buffer> edep_squared_; /*!< Buffer storing energy deposit squared on OpenCL device */
  std::shared_ptr<cl::Buffer> hit_; /*!< Buffer storing hit on OpenCL device */
  std::shared_ptr<cl::Buffer> photon_tracking_; /*!< Buffer storing photon tracking on OpenCL device */
} GGEMSDoseRecording; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSDOSERECORDING_HH
