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

#ifndef __OPENCL_C_VERSION__

#include <memory>

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

#endif

#ifdef __OPENCL_C_VERSION__

#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/geometries/GGEMSGeometryConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline void dose_photon_tracking(global GGEMSDoseParams* dose_params, global GGint* photon_tracking, GGfloat3 const* position)
{
  // Check position of photon inside dosemap limits
  if (position->x < dose_params->border_min_xyz_.x + EPSILON6 || position->x > dose_params->border_max_xyz_.x - EPSILON6) return;
  if (position->y < dose_params->border_min_xyz_.y + EPSILON6 || position->y > dose_params->border_max_xyz_.y - EPSILON6) return;
  if (position->z < dose_params->border_min_xyz_.z + EPSILON6 || position->z > dose_params->border_max_xyz_.z - EPSILON6) return;

  // Get index in dose map
  GGint3 dosel_id = convert_int3((*position - dose_params->border_min_xyz_) * dose_params->inv_size_of_dosels_);
  GGint global_dosel_id = dosel_id.x + dosel_id.y * dose_params->number_of_dosels_.x + dosel_id.z * dose_params->number_of_dosels_.x * dose_params->number_of_dosels_.y;
  atomic_add(&photon_tracking[global_dosel_id], 1);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn
  \param dose_params - params associated to dosemap
  \param
  \brief Recording data for dosimetry
*/
inline void dose_record_standard(global GGEMSDoseParams* dose_params, global GGdouble* edep_tracking, global GGdouble* edep_squared_tracking, global GGint* hit_tracking, GGfloat edep, GGfloat3 const* position)
{
  // Check position of photon inside dosemap limits
  if (position->x < dose_params->border_min_xyz_.x + EPSILON6 || position->x > dose_params->border_max_xyz_.x - EPSILON6) return;
  if (position->y < dose_params->border_min_xyz_.y + EPSILON6 || position->y > dose_params->border_max_xyz_.y - EPSILON6) return;
  if (position->z < dose_params->border_min_xyz_.z + EPSILON6 || position->z > dose_params->border_max_xyz_.z - EPSILON6) return;

  // Get index in dose map
  GGint3 dosel_id = convert_int3((*position - dose_params->border_min_xyz_) * dose_params->inv_size_of_dosels_);
  GGint global_dosel_id = dosel_id.x + dosel_id.y * dose_params->number_of_dosels_.x + dosel_id.z * dose_params->number_of_dosels_.x * dose_params->number_of_dosels_.y;

  atomic_add(&hit_tracking[global_dosel_id], 1);

  // ggems_atomic_add_f64( dose->edep, index_phantom.w, f64( Edep ) );
  // ggems_atomic_add_f64( dose->edep_squared, index_phantom.w, f64( Edep) * f64( Edep ) );
}

#endif

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSDOSERECORDING_HH
